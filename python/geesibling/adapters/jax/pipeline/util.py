
import functools
from collections import OrderedDict
from typing import  Dict, Sequence, Any
import jax
from jax import linear_util as lu
from jax._src.source_info_util import SourceInfo
from jax._src.api import ShapeDtypeStruct
from jax.interpreters import partial_eval as pe
from jax import core
from jax.interpreters import xla, mlir
from jax.core import (Atom, ClosedJaxpr, Jaxpr, JaxprEqn, Primitive, Var, DropVar, Literal, AbstractValue, ShapedArray)
from jax.tree_util import tree_flatten
from jax.api_util import shaped_abstractify
from flax.training import train_state


from jax._src.util import wrap_name
import itertools as it
from jax._src import dispatch
from jax._src import source_info_util
from jax._src.lib import xla_extension as xe
import logging
from jax import lax
from jax.lib import xla_client as xc, xla_bridge as xb
from geesibling.adapters.jax.pipeline.wrapped_hlo import WrappedHlo

logger = logging.getLogger(__name__)

def auto_static_argnums(args: Sequence[Any]):
    """Return the indices of static arguments according to heuristic rules."""

    def is_static_arg(arg):
        # 检查arg是否为bool,int,float,str类型
        if isinstance(arg, (bool, int, float, str)):
            return True
        # 若arg与训练有关则应该是动态参数
        if isinstance(arg, train_state.TrainState):
            return False
        

        # 把arg参数展平后，判断arg里所有的元素是否都不是静态参数，若有一个为静态参数，则整体都为静态参数
        xs, _ = tree_flatten(arg)
        for x in xs:
            try:
                x = shaped_abstractify(x)
            except TypeError:
                return True
        return False

    return tuple(i for i in range(len(args)) if is_static_arg(args[i]))


# 将参数x转为抽象参数
def abstractify_with_aval(x):
    if isinstance(x, ShapedArray):
        return x
    elif isinstance(x, ShapeDtypeStruct):
        return ShapedArray(x.shape, x.dtype, named_shape=x.named_shape)
    else:
        return xla.abstractify(x)


class GradFuncTransformContext:
    """
    A context to hold transformations applied to the forward function
    before calling alpa.grad or alpa.value_and_grad.
    """
    transforms = []

    def __init__(self, transform):
        self.transform = transform

    def __enter__(self):
        GradFuncTransformContext.transforms.append(self.transform)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        GradFuncTransformContext.transforms.pop()


class OrderedSet:
    """An ordered set implemented by using the built-in OrderedDict."""

    def __init__(self, iterable=()):
        self.dict = OrderedDict()
        self.dict.update({x: None for x in iterable})

    def add(self, *args):
        self.dict.update({x: None for x in args})

    def update(self, other):
        self.dict.update({x: None for x in other})

    def union(self, other):
        result = OrderedSet(self)
        result.update(other)
        return result

    def intersection_update(self, other):
        for x in [x for x in self.dict if x not in other]:
            del self.dict[x]

    def intersection(self, other):
        return OrderedSet(x for x in self if x in other)

    def discard(self, element):
        if element in self:
            del self.dict[element]

    def remove(self, element):
        if element not in self:
            raise KeyError(element)
        del self.dict[element]

    def clear(self):
        self.dict.clear()

    def difference(self, other):
        return OrderedSet([x for x in self if x not in other])

    def difference_update(self, other):
        for x in other:
            self.discard(x)

    def symmetric_difference(self, other):
        result = OrderedSet()
        for x in self:
            if x not in other:
                result.add(x)
        for x in other:
            if x not in self:
                result.add(x)
        return result

    def __iter__(self):
        return iter(self.dict)

    def __len__(self):
        return len(self.dict)

    def __contains__(self, element):
        return element in self.dict

    def __repr__(self):
        return "OrderedSet([" + ", ".join(repr(x) for x in self) + "])"

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __ior__(self, other):
        self.update(other)

    def __iand__(self, other):
        self.intersection_update(other)

    def __isub__(self, other):
        self.difference_update(other)

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return self.dict == other.dict
        return False

    @classmethod
    def __class_getitem__(cls, item):
        return f"{cls.__name__}[{item.__name__}]"
    

def cached_property(fn, *args, **kwargs):
    """
    Decorator to make a function a "cached property".

    This means that it is a property whose return value is cached after the
    first time it is called.

    Args:
        fn: The function to be made a cached property
        *args: Any args for the function
        **kwargs: Any kwargs for the function
    Returns:
        function
    """
    return property(functools.lru_cache()(fn, *args, **kwargs))







    


########################################
##### Jaxpr Utilities
########################################
def clone_jaxpr(closed_jaxpr: ClosedJaxpr,
                invars: Sequence[Atom] = None,
                outvars: Sequence[Var] = None,
                eqns: Sequence[JaxprEqn] = None,
                constvars: Sequence[Var] = None,
                consts: Sequence = None):
    """Clone a jaxpr and replace members if they are provided."""
    constvars = closed_jaxpr.jaxpr.constvars if constvars is None else constvars
    invars = closed_jaxpr.jaxpr.invars if invars is None else invars
    outvars = closed_jaxpr.jaxpr.outvars if outvars is None else outvars
    eqns = closed_jaxpr.jaxpr.eqns if eqns is None else eqns
    consts = closed_jaxpr.consts if consts is None else consts
    jaxpr = Jaxpr(constvars, invars, outvars, eqns)
    return ClosedJaxpr(jaxpr, consts)


def new_jaxpr_eqn(invars,
                  outvars,
                  primitive,
                  params,
                  effects=None,
                  source_info=None):
    """Create a new jaxpr equation."""
    effects = effects or core.no_effects
    return core.new_jaxpr_eqn(invars, outvars, primitive, params, effects,
                              source_info)


def clone_jaxpr_eqn(eqn: JaxprEqn,
                    invars: Sequence[Atom] = None,
                    outvars: Sequence[Var] = None,
                    primitive: Primitive = None,
                    params: Dict[str, Any] = None,
                    effects: Any = None,
                    source_info: SourceInfo = None):
    invars = list(invars or eqn.invars)
    outvars = list(outvars or eqn.outvars)
    primitive = primitive or eqn.primitive
    params = dict(params or eqn.params)
    source_info = source_info or eqn.source_info
    effects = effects or eqn.effects
    return new_jaxpr_eqn(invars, outvars, primitive, params, effects,
                         source_info)


# make sliced_jaxprs by sliced_eqns
def slices_to_jaxpr(
        closed_jaxpr: ClosedJaxpr,
        sliced_eqns: Sequence[Sequence[JaxprEqn]]) -> Sequence[ClosedJaxpr]:
    """Wrap sliced equations to a list of ClosedJaxpr."""
    n_eqns = len(sliced_eqns)
    global_invars = OrderedSet(closed_jaxpr.jaxpr.invars)
    global_outvars = OrderedSet(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    layer_invars = [OrderedSet() for _ in range(n_eqns)]
    layer_outvars = [OrderedSet() for _ in range(n_eqns)]
    layer_consts = [{} for _ in range(n_eqns)]

    var_layer_dict = {}  # Dict[var -> layer_idx]
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            #print("============================================")
            #print("eqn invars:",eqn.invars)
            #print("eqn outvars:",eqn.outvars)
            for var in eqn.invars:
                if isinstance(var, Literal):
                    continue
                if var in global_consts:
                    layer_consts[i][var] = global_consts[var]
                elif var in global_invars:
                    layer_invars[i].add(var)
                elif var_layer_dict[var] != i:
                    layer_invars[i].add(var)
                    layer_outvars[var_layer_dict[var]].add(var)
                else:
                    assert var_layer_dict[var] == i
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i
                if var in global_outvars:
                    layer_outvars[i].add(var)

    result = []
    for i, eqns in enumerate(sliced_eqns):
        new_jaxpr = Jaxpr(list(layer_consts[i].keys()), list(layer_invars[i]),
                          list(layer_outvars[i]), eqns)
        new_closed_jaxpr = ClosedJaxpr(new_jaxpr,
                                       list(layer_consts[i].values()))
        result.append(new_closed_jaxpr)
    return result




def get_var_mapping(mapping, var):
    """map the var to a new value if var is Var and in the mapping."""
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var
    

def trace_jaxpr_with_micro_batch(fun: lu.WrappedFun,
                                 batch_invars: Sequence[bool],
                                 num_micro_batches: int,
                                 raw_avals: Sequence[AbstractValue],
                                 batch_dim: int = 0):
    """Trace the jaxpr of the computation of a micro batch."""
    assert batch_dim == 0, "Only support batch_dim == 0"


    avals = []
    batch_size = None
    for aval, is_batch_var in zip(raw_avals, batch_invars):
        if is_batch_var:
            assert aval.shape[0] % num_micro_batches == 0, (
                f"The batch size must be divisable by num_micro_batches. "
                f"batch_size = {aval.shape[0]}, "
                f"num_micro_batches = {num_micro_batches}")
            if batch_size is None:
                batch_size = aval.shape[0] // num_micro_batches
            else:
                assert batch_size == aval.shape[0] // num_micro_batches, (
                    "The batch dimension must be the same for all batch vars.")
            shape = (batch_size,) + aval.shape[1:]
            avals.append(aval.update(shape=shape))
        else:
            avals.append(aval)
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)

    return closed_jaxpr, batch_size



def prefetch(x):
  if isinstance(x, device_array.DeviceArray):
    x.copy_to_host_async()
  return x

def jaxpr_literals(jaxpr):
  """Generates all the literals inside a jaxpr, including nested subjaxprs."""
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if type(v) is core.Literal:
        yield v.val
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_literals(subjaxpr)

def jaxpr_to_hlo(name: str,
                 closed_jaxpr: ClosedJaxpr,
                 donated_invars: Sequence[bool],
                 platform: str = "cuda"):
    """Convert a jaxpr to a wrapped XLA HloModule.

    Reference code: jax/jax/_src/dispatch.py::lower_xla_callable
    """
    consts = closed_jaxpr.consts
    map(prefetch,
        it.chain(consts, jaxpr_literals(closed_jaxpr.jaxpr)))

    # Convert jaxpr to XLA HLO
    tuple_args = False
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())
    name_stack = source_info_util.new_name_stack(wrap_name(name, "parallelize"))
    closed_jaxpr = ClosedJaxpr(closed_jaxpr.jaxpr, consts)
    ordered_effects = [
        eff for eff in closed_jaxpr.effects if eff in core.ordered_effects
    ]
    lowering_result = mlir.lower_jaxpr_to_module(
        name, closed_jaxpr, ordered_effects=ordered_effects, backend_or_name=None, 
        platforms=[platform], axis_context=mlir.ReplicaAxisContext(axis_env), 
        name_stack=name_stack, donated_args=donated_invars, 
        lowering_parameters=mlir.LoweringParameters())
    xla_computation = xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(lowering_result.module),
        use_tuple_args=tuple_args,
        return_tuple=True)
    return WrappedHlo(xla_computation)



_DISABLE_NUMBA = False

def maybe_numba_jit(func):
    """Decorator to mark a function as numba jitted if numba is available."""
    try:
        from numba import jit  # pylint: disable=import-outside-toplevel
        jitted_func = jit(nopython=True)(func)

        def wrapper(*args, **kwargs):
            if _DISABLE_NUMBA:
                return func(*args, **kwargs)
            return jitted_func(*args, **kwargs)

        return wrapper
    except ImportError:
        logger.warning("Install numba to jit and accelerate the function.")
        return func
    


#==========================================
# JAX eqns evaluate
#==========================================

non_trivial_primitive = [lax.dot_general_p, lax.conv_general_dilated_p]


def eqn_flops(eqn: JaxprEqn) -> float:
    """Get the FLOP of a jaxpr equation."""
    if "jaxpr" in eqn.params:
        return sum(eqn_flops(x) for x in eqn.params["jaxpr"].eqns)

    if eqn.primitive not in non_trivial_primitive:
        return 0

    new_inv = [inv for inv in eqn.invars if isinstance(inv, Var)]
    jaxpr = Jaxpr([], new_inv, eqn.outvars, [eqn])
    closed_jaxpr = ClosedJaxpr(jaxpr, [])
    hlo_module = jaxpr_to_hlo("tmp", closed_jaxpr, [
        False,
    ] * len(jaxpr.invars)).get_module()

    backend = xb.get_backend("cpu")
    properties = xc._xla.hlo_module_cost_analysis(  # pylint: disable=protected-access
        backend, hlo_module)
    return properties["flops"] if "flops" in properties else 0.0


def heavy_count(eqn):
    """Check the number of heavy ops in the eqn."""
    if "jaxpr" in eqn.params:
        return sum(heavy_count(x) for x in eqn.params["jaxpr"].eqns)

    if eqn.primitive not in non_trivial_primitive:
        return 0
    return 1


def is_nontrivial(eqn):
    """Check if the eqn is nontrivial."""
    return heavy_count(eqn) > 0


