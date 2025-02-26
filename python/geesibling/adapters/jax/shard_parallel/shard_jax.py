from functools import partial
import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Var, new_jaxpr_eqn,Literal

def shard_parallel(jaxpr, params, out_tree):
    print("jaxpr=======================")
    print(jaxpr)
    print("jaxprs======================")

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh((2, 1), devices[:2])
    mesh = Mesh(device_mesh, axis_names=('x','y'))


    @partial(shard_map, mesh=mesh, in_specs=(P(None, None), P(None, 'x')), out_specs=P(None, 'x'))
    def parallel_dot_general(A, B):
        print(f"input shard shape: {A.shape}, {B.shape}")
        return jax.lax.dot_general(A, B, dimension_numbers=(((1,), (0,)), ((), ())), precision=None)
    
    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'dot_general':
            if eqn.params['dimension_numbers'] == (((1,), (0,)), ((), ())):
                print("dot_general=======================")
                gensym_fn = gensym(jaxpr.jaxpr)
                new_outvars = [gensym_fn(var.aval) for var in eqn.outvars]
                new_eqn = new_jaxpr_eqn(
                    invars=eqn.invars,
                    outvars=new_outvars,
                    primitive=parallel_dot_general,
                    params=eqn.params
                )
                print(new_eqn) 
                new_eqns.append(new_eqn)
        else:
            new_eqns.append(eqn)
    
    print(new_eqns)

    new_jaxpr = ClosedJaxpr(jaxpr.constvars, new_eqns, jaxpr.outvars)

    print("new_jaxpr=======================")
    print(new_jaxpr)
    print("new_jaxprs======================")

    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)
    # result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

    return result