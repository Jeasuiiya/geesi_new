import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Jaxpr, new_jaxpr_eqn, Primitive
from functools import partial,lru_cache
from geesibling.adapters.jax.pipeline.util import eqn_flops

dynamic_threshold = 1e7
devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((2, ), devices[:2])
mesh = Mesh(device_mesh, axis_names=('x',))

dot_general_shard_primitive = Primitive('dot_general_shard')

def shard_data(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(data, sharding)

@lru_cache(maxsize=None)
def get_compiled_fn(mesh, in_specs, out_specs, dimension_numbers):
    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def _fn(a, b):
        return jax.lax.dot_general(a, b, dimension_numbers=dimension_numbers)
    return jax.jit(_fn)

def dot_general_shard_impl(*args, mesh, in_specs, out_specs,flag, dimension_numbers):
    if not flag:
        args = [shard_data(arg, mesh, spec) for arg, spec in zip(args, in_specs)]
    parallel_fn = get_compiled_fn(mesh, in_specs, out_specs, dimension_numbers)
    return parallel_fn(*args)

dot_general_shard_primitive.def_impl(dot_general_shard_impl)

dimension_numbers_map = {
    "(((0,), (0,)), ((), ()))": ((P(None, 'x'), P(None, None)), P('x', None), False),
    "(((1,), (0,)), ((), ()))": ((P('x', None), P(None, None)), P('x', None), True),
    "(((2,), (0,)), ((), ()))": ((P(None, 'x', None), P(None, None)), P(None, 'x', None), True),
    "(((1,), (3,)), ((0, 2), (0, 1)))": ((P(None, None, None, None), P(None, None, 'x', None)), P(None, None, 'x', None), True),
    "(((3,), (3,)), ((0, 2), (0, 2)))": ((P(None, 'x', None, None), P(None, None, None, None)), P(None, None, 'x', None), True),
    "(((0, 1), (0, 1)), ((), ()))": ((P(None, None, 'x'), P(None, None, None)), P('x', None), False),
}

def shard_parallel(jaxpr, params, out_tree):    
    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            dimension_numbers=eqn.params["dimension_numbers"]
            if str(dimension_numbers) not in dimension_numbers_map:
                new_eqns.append(eqn)
            else:
                flops = eqn_flops(eqn)
                if flops < dynamic_threshold:
                    new_eqns.append(eqn)
                else:
                    in_specs, out_specs, flag = dimension_numbers_map[str(dimension_numbers)]
                    new_eqn = new_jaxpr_eqn(
                        invars=eqn.invars,
                        outvars=eqn.outvars,
                        primitive=dot_general_shard_primitive,
                        params={
                            "mesh": mesh,
                            "flag":flag,
                            "in_specs": in_specs,
                            "out_specs": out_specs,
                            "dimension_numbers": dimension_numbers
                        },
                        effects=set()
                    )
                    new_eqns.append(new_eqn)
        else:
            new_eqns.append(eqn)

    new_jaxpr_core = Jaxpr(jaxpr.jaxpr.constvars, jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars, new_eqns)
    new_jaxpr = ClosedJaxpr(new_jaxpr_core, jaxpr.consts)
    
    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)
    result = jax.device_put(result, devices[0])

    return result
