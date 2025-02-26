import collections
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple
import itertools as it
import jax
import jax.numpy as jnp
import jax.core as jcore
from geesibling.adapters.jax.schedule import GraphPortRef
from geesibling.adapters.jax.profile import profile_eqn, profile
from geesibling.core.types import Graph, Node
from geesibling.core.lib._graph import DataType
from geesibling.tools import log
import ml_dtypes
import numpy as np

__doc__ = """
convert jaxpr to core graph
Author: yiguangzheng
datetime: 2023.1.9
version: 1 2023.1.9 first commit
"""
bfloat16 = ml_dtypes.bfloat16
float8_e4m3fn = ml_dtypes.float8_e4m3fn
float8_e5m2 = ml_dtypes.float8_e5m2

SHAPE_ARRAY_DTYPE_TO_DATA_TYPE = {
    np.dtype("bool"): DataType.BOOL,
    np.dtype("int8"): DataType.I8,
    np.dtype("int16"): DataType.I16,
    np.dtype("int32"): DataType.I32,
    np.dtype("int64"): DataType.I64,
    np.dtype("uint8"): DataType.U8,
    np.dtype("uint16"): DataType.U16,
    np.dtype("uint32"): DataType.U32,
    np.dtype("uint64"): DataType.U64,
    np.dtype(float8_e4m3fn): DataType.F8E4M3FN,
    np.dtype(float8_e5m2): DataType.F8E5M2,
    np.dtype(bfloat16): DataType.BF16,
    np.dtype("float16"): DataType.F16,
    np.dtype("float32"): DataType.F32,
    np.dtype("float64"): DataType.F64,
}


@dataclass
class GraphWrapper:
    graph: Graph
    params: Dict  # node params
    invars: List[GraphPortRef]
    returns: List[GraphPortRef]
    node_output_type: Dict[Tuple[str, int], Any]
    node_input_type: Dict[Tuple[str, int], Any]
    node_ref_const: Dict[Tuple[str, int], Any]


class ConvertContext:
    """
    hold some map using in coversion and provide someo helper for conversion
    """

    var_ids: DefaultDict[jcore.Var, int]
    id_vars: Dict[int, jcore.Var]
    var_outputs: Dict[jcore.Atom, Tuple[Node, int]]
    literal_inputs = Dict[str, Node]
    node_name_record = {}
    id_params = {}
    # record opaque type for node output
    node_output_type: Dict[GraphPortRef, Any]
    node_input_type: Dict[GraphPortRef, Any]

    def __init__(self):
        self.var_ids = collections.defaultdict(it.count().__next__, {})
        self.id_vars = {}
        self.var_outputs = {}
        self.literal_inputs = {}
        self.node_name_record = {}
        self.node_output_type = {}
        self.node_input_type = {}

    def register_var(self, var: jcore.Var):
        """
        register var and id such as input, output, const and SSA var in maps
        """
        var_id = self.var_ids[var]
        self.id_vars[var_id] = var

    def register_output(self, var, node, index):
        """
        register node's output.
        an output of a node can be search by a var.
        """
        self.var_outputs[var] = (node, index)

    def gen_name(self, prefix):
        """
        generate node name
        """
        if self.node_name_record.get(prefix, None) is None:
            index = 0
        else:
            index = self.node_name_record[prefix] + 1
        self.node_name_record[prefix] = index
        return f"{prefix}_{index}"


def process_var(v: jcore.Var, context: ConvertContext, name_prifix=None):
    """
    process vars of jaxpr.
    the var will be register as var and output in context.
    """
    if isinstance(v, (jcore.Literal, jcore.DropVar)):
        if context.literal_node.get(v.val) is None:
            node = Node(context.gen_name("Literal"), "Literal")
            node.attrs["value"] = v.val

            return node
        return context.literal_node[v.val]
    if context.var_outputs.get(v) is None:
        context.register_var(v)
        node = Node(context.gen_name(name_prifix), name_prifix)
        dtype = v.aval.dtype
        if dtype not in SHAPE_ARRAY_DTYPE_TO_DATA_TYPE:
            context.node_output_type[GraphPortRef(node.name, 0)] = dtype
            dtype = DataType.Other
        else:
            dtype = SHAPE_ARRAY_DTYPE_TO_DATA_TYPE[dtype]
        node.add_outputport(dtype, v.aval.shape, 0)
        context.register_output(v, node, 0)
        return node
    return context.var_outputs[v]


def process_vars(vs: Sequence[Any], context: ConvertContext, name_prifix=None):
    """
    register vars
    """
    return [process_var(i, context, name_prifix) for i in vs]


def process_literal_invars(invars, in_node, context: ConvertContext):
    """
    process eqn input vars.
    """
    for i, v in enumerate(invars):
        if isinstance(v, (jcore.Literal, jcore.DropVar)):
            node = Node(context.gen_name("Const"), "Const")
            # node.attrs["value"] = str(v.val)
            val = np.array(v.val)
            node.add_outputport(SHAPE_ARRAY_DTYPE_TO_DATA_TYPE[val.dtype], val.shape, 0, val)
            context.literal_inputs[(in_node, i)] = node


def _add_abstract_output(context, var, node, index):
    dtype = var.aval.dtype
    if dtype not in SHAPE_ARRAY_DTYPE_TO_DATA_TYPE:
        context.node_output_type[GraphPortRef(node.name, index)] = dtype
        dtype = DataType.Other
    else:
        dtype = SHAPE_ARRAY_DTYPE_TO_DATA_TYPE[dtype]
    node.add_outputport(dtype, var.aval.shape, index)


def _add_abstract_input(context, var, node, index, ref_name, ref_index):
    dtype = var.aval.dtype
    if dtype not in SHAPE_ARRAY_DTYPE_TO_DATA_TYPE:
        context.node_input_type[GraphPortRef(node.name, index)] = dtype
        dtype = DataType.Other
    else:
        dtype = SHAPE_ARRAY_DTYPE_TO_DATA_TYPE[dtype]
    node.add_inputport(ref_name, ref_index, index, dtype, var.aval.shape)

    # node.add_outputport(dtype, var.aval.shape, index)


def process_eqn(eqn, context: ConvertContext):
    """
    process eqn.
    every eqn will be register as a node in graph.
    """
    _ = [context.register_var(var) for var in eqn.outvars]
    node = Node(context.gen_name(eqn.primitive.name), eqn.primitive.name)
    context.id_params[node.name] = eqn.params  # todo(yiguangzheng): recursive jaxpr
    for i, var in enumerate(eqn.outvars):
        context.register_output(var, node, i)
        _add_abstract_output(context, var, node, i)

    process_literal_invars(eqn.invars, node, context)

    def build_inputs(index, var):
        if isinstance(var, jcore.Literal):
            previous_node = context.literal_inputs[(node, index)]
            ref_name = previous_node.name
            ref_index = 0
        else:
            previous_node = context.var_outputs[var][0]
            ref_name = previous_node.name
            ref_index = context.var_outputs[var][1]

        # build previous node output to current node connect
        previous_node.add_output(node.name)
        _add_abstract_input(context, var, node, index, ref_name, ref_index)
        # node.add_inputport(ref_name, ref_index, index, SHAPE_ARRAY_DTYPE_TO_DATA_TYPE[var.aval.dtype], var.aval.shape)
        node.add_input(ref_name)

    _ = [build_inputs(i, v) for i, v in enumerate(eqn.invars)]
    compute, input_memory, output_memory = profile_eqn(eqn)
    # compute, input_memory, output_memory = 100,100,100
    node.compute_cost = compute
    node.input_memory = input_memory
    node.output_memory = output_memory
    node.persistent_memory = output_memory
    return node


def process_eqns(eqns, context: ConvertContext):
    """
    process eqns
    """
    return [process_eqn(i, context) for i in eqns]


def process_output(outvar, context: ConvertContext):
    """
    process outputs of jaxpr.
    """
    node_name = context.var_outputs[outvar][0].name
    index = context.var_outputs[outvar][1]
    return (node_name, index)


def process_outputs(outvars, context: ConvertContext):
    """
    process outputs.
    """
    return [process_output(i, context) for i in outvars]


def jaxpr2graph(jaxpr: jcore.ClosedJaxpr):
    """
    convert jax to geesibling graph
    Args:
        jaxpr: a ClosedJaxpr that will be converted
    """
    context = ConvertContext()
    graph = Graph()
    log.trace("jaxpr: %s", jaxpr.jaxpr)
    log.trace("constvars: %s", jaxpr.jaxpr.constvars)
    input_nodes = process_vars(jaxpr.jaxpr.invars, context, "Input")
    const_nodes = process_vars(jaxpr.jaxpr.constvars, context, "ConstVar")
    node_ref_const = {}
    for value, var in zip(jaxpr.consts, jaxpr.jaxpr.constvars):
        key = (context.var_outputs[var][0].name, context.var_outputs[var][1])
        node_ref_const[key] = value
    log.trace("const nodes: %s", const_nodes)
    with profile():
        eqn_nodes = process_eqns(jaxpr.jaxpr.eqns, context)
    outputs = process_outputs(jaxpr.jaxpr.outvars, context)

    _ = [graph.add_node(n) for n in input_nodes]
    _ = [graph.add_node(n) for n in const_nodes]
    _ = [graph.add_node(n) for n in context.literal_inputs.values()]
    _ = [graph.add_node(n) for n in eqn_nodes]
    _ = [graph.add_return(o) for o in outputs]

    input_vars = list(
        map(lambda x: GraphPortRef(context.var_outputs[x][0].name, context.var_outputs[x][1]), jaxpr.jaxpr.invars)
    )
    return GraphWrapper(graph, context.id_params, input_vars, graph.returns, context.node_output_type, context.node_input_type, node_ref_const)


def add(x, y):
    a = x + 1
    b = a * 2
    return b + y


def add1(x, y):
    a = x * y
    return x + a


def selu(x, alpha=1.67, lmbda=1.05):
    y = jnp.arange(10)
    z = x + y
    return lmbda * jax.numpy.where(x > 0, z, alpha * jax.numpy.exp(z) - alpha)
