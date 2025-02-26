from functools import partial
import jax
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.core import JaxprEqn, Literal

def shard_parallel(jaxpr, params, out_tree):
    print("jaxpr=================================")
    print(jaxpr)
    print("jaxprs================================")

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh((2, 2), devices[:4])
    mesh = Mesh(device_mesh, axis_names=('x','y'))

    @partial(shard_map, mesh=mesh, in_specs=(P('x', None), P(None, 'y')), out_specs=P('x', 'y'))
    def parallel_dot_general(A, B):
        print(f"input shard shape: {A.shape}, {B.shape}")
        return jax.lax.dot_general(A, B, dimension_numbers=(((1,), (0,)), ((), ())), precision=None)
    # parallel_dot_general = pjit(parallel_dot_general, in_specs=(P('x', None), P(None, 'y')), out_specs=P('x', 'y'))

    def execute_parallel(jaxpr, params):
        env = {}
        for var, val in zip(jaxpr.jaxpr.invars, params):
            env[var] = val

        if not jaxpr.jaxpr.eqns:
            result = [env[var] for var in jaxpr.jaxpr.invars]
        else:
            for eqn in jaxpr.jaxpr.eqns:
                invals = []
                for var in eqn.invars:
                    if isinstance(var, Literal):
                        invals.append(var.val)
                    else:
                        invals.append(env[var])

                func = eqn.primitive

                if func.name == 'dot_general':
                    dimension_numbers = eqn.params['dimension_numbers']
                    # for i, array in enumerate(invals):
                    #     print(f"inval {i} shape: {array.shape}")
                    if dimension_numbers == (((1,), (0,)), ((), ())):
                        outvals = parallel_dot_general(*invals)
                        print(f"Output result shape: {outvals.shape}")
                        # for device in outvals.devices():
                        #     print(f"result buffer on device: {device}")
                        outvals=jax.device_put(outvals, devices[0])
                    else:
                        outvals = func.bind(*invals, **eqn.params)
                elif func.name == 'custom_jvp_call':
                    call_jaxpr = eqn.params['call_jaxpr']
                    jvp_jaxpr_thunk = eqn.params['jvp_jaxpr_thunk']
                    num_consts = eqn.params['num_consts']
                    symbolic_zeros = eqn.params['symbolic_zeros']
                    sub_result = execute_parallel(call_jaxpr, invals)

                    if isinstance(sub_result, list):
                        sub_result = jax.numpy.stack(sub_result)
                    elif isinstance(sub_result, tuple):
                        sub_result = jax.numpy.stack(sub_result)
                    
                    outvals = jax.numpy.squeeze(jax.numpy.array(sub_result))
                    print(outvals.shape)
                else:
                    outvals = func.bind(*invals, **eqn.params)


                if not isinstance(outvals, tuple):
                    outvals = (outvals,)
                for var, val in zip(eqn.outvars, outvals):
                    env[var] = val

            result = [env[var] for var in jaxpr.jaxpr.outvars]

            for i, array in enumerate(result):
                print(f"result {i} shape: {array.shape}")            
        return result

    return execute_parallel(jaxpr, params)