import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Jaxpr, new_jaxpr_eqn, Primitive
from functools import partial,lru_cache
from geesibling.adapters.jax.pipeline.util import eqn_flops

dynamic_threshold = 1e7

dot_general_shard_primitive = Primitive('dot_general_shard')

def shard_data(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(data, sharding)

@lru_cache(maxsize=None)
def get_compiled_fn(mesh, in_specs, out_specs, dimension_numbers, precision, preferred_element_type):
    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def _fn(a, b):
        result = jax.lax.dot_general(
            a, b, dimension_numbers=dimension_numbers, 
            **{k: v for k, v in {"precision": precision, "preferred_element_type": preferred_element_type}.items() if v is not None}
        )
        return result
    return jax.jit(_fn)

def dot_general_shard_impl(*args, mesh, in_specs, out_specs,flag, dimension_numbers,precision, preferred_element_type):
    if not flag:
        args = [shard_data(arg, mesh, spec) for arg, spec in zip(args, in_specs)]
    parallel_fn = get_compiled_fn(mesh, in_specs, out_specs, dimension_numbers, precision, preferred_element_type)
    return parallel_fn(*args)

dot_general_shard_primitive.def_impl(dot_general_shard_impl)

dimension_numbers_map = {
    "(((0,), (0,)), ((), ()))": ((P(None, 'x'), P(None, 'y')), P('x', 'y'), False),
    "(((1,), (0,)), ((), ()))": ((P('x', None), P(None, 'y')), P('x', 'y'), True),
    "(((2,), (0,)), ((), ()))": ((P(None, 'x', None), P(None, 'y')), P(None, 'x', 'y'), True),
    "(((1,), (3,)), ((0, 2), (0, 1)))": ((P(None, None, None, 'x'), P(None, None, 'y', None)), P(None, None, 'y', 'x'), True),
    "(((3,), (3,)), ((0, 2), (0, 2)))": ((P(None, 'x', None, None), P(None, 'y', None, None)), P(None, None, 'x', 'y'), True),
    "(((0, 1), (0, 1)), ((), ()))": ((P(None, None, 'x'), P(None, None, 'y')), P('x', 'y'), False),
}

def get_best_sharding(dim1, dim2, tp_num):
    for new_tp in range(tp_num , 0, -1):
        factors = [(i, new_tp // i) for i in range(1, new_tp + 1) if new_tp % i == 0]
        factors.sort(key=lambda p: (abs(p[0] - p[1]), -p[0]))
        for x, y in factors:
            if dim1 % x == 0 and dim2 % y == 0:
                return (x, y)
    return (1, 1)

def shard_parallel(jaxpr, params, tp_num, out_tree):  
    # print("jaxpr=======================")
    # print(jaxpr)
    # print("jaxpr=======================")
    devices = jax.devices() 
    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            dimension_numbers=eqn.params["dimension_numbers"]
            precision = eqn.params.get("precision", None)
            preferred_element_type = eqn.params.get("preferred_element_type", None)
            if str(dimension_numbers) not in dimension_numbers_map:
                new_eqns.append(eqn)
            else:
                flops = eqn_flops(eqn)
                if flops < dynamic_threshold:
                    new_eqns.append(eqn)
                else:
                    in_specs, out_specs, flag = dimension_numbers_map[str(dimension_numbers)]
                    x_dim, y_dim = eqn.invars[0].aval.shape[in_specs[0].index('x')], eqn.invars[1].aval.shape[in_specs[1].index('y')]
                    best_x, best_y = get_best_sharding(x_dim, y_dim, tp_num)
                    device_mesh = mesh_utils.create_device_mesh((best_x, best_y), devices[:tp_num])
                    mesh = Mesh(device_mesh, axis_names=('x', 'y'))
                    
                    new_eqn = new_jaxpr_eqn(
                        invars=eqn.invars,
                        outvars=eqn.outvars,
                        primitive=dot_general_shard_primitive,
                        params={
                            "mesh": mesh,
                            "flag":flag,
                            "in_specs": in_specs,
                            "out_specs": out_specs,
                            "dimension_numbers": dimension_numbers,
                            "precision": precision,
                            "preferred_element_type": preferred_element_type
                        },
                        effects=set()
                    )
                    new_eqns.append(new_eqn)
        else:
            new_eqns.append(eqn)

    new_jaxpr_core = Jaxpr(jaxpr.jaxpr.constvars, jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars, new_eqns)
    new_jaxpr = ClosedJaxpr(new_jaxpr_core, jaxpr.consts)
    
    # print("new_jaxpr=======================")
    # print(new_jaxpr)
    # print("new_jaxpr=======================")

    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)
    result = jax.device_put(result, devices[0])
    # result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

    return result
