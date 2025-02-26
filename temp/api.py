import logging
import functools
from typing import Callable, Optional
import jax
from jax.tree_util import tree_flatten
from jax.api_util import argnums_partial, donation_vector,flatten_fun_nokwargs
from jax._src import api
from geesibling.core.lib._graph import Device
from geesibling.adapters.jax.pipeline.devicecontext import init_global_cluster,get_global_virtual_physical_mesh,get_sliced_virtual_submeshes,init_global_cluster
from geesibling.adapters.jax.model_parallelism import MakeScheduleContext
from geesibling.adapters.jax.pipeline.layer_construction import layer_level_transformation
from geesibling.adapters.jax.pipeline.primitive_def import mark_gradient
from geesibling.adapters.jax.pipeline.util import auto_static_argnums, abstractify_with_aval, trace_jaxpr_with_micro_batch
from geesibling.tools import log
import ray.util.collective as col
from jax import linear_util as lu
from jax._src.maps import FrozenDict
import ray
import time
from geesibling.adapters.jax.pipeline.instructions import PipelineInstType
__doc__ = """
parallelize api
Author: yiguangzheng
datetime: 2023.7.4
version: 1 2023.7.4 first commit
"""

DEVICE_MAP = {"": jax.devices("cpu")[0] if len(jax.devices("gpu")) == 0 else jax.devices("gpu")[0]}

from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
def register_device():
    for i in jax.devices("cpu"):
        DEVICE_MAP[str(i)] = i
    for i in jax.devices("gpu"):
        DEVICE_MAP[str(i)] = i
    log.debug(DEVICE_MAP)


register_device()
logging.basicConfig(level=logging.WARNING)

def device_config(attrs):
    d = []
    for k, v in attrs.items():
        d.append(Device(v["type"], k, v["memory"], v["free_memory"], v["execute_time"]))
    return d

#抽象的数据转换
def _abstractify(args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    return map(jax.api_util.shaped_abstractify, flat_args), flat_args, in_tree#将其转化为含有形状和数据类型等信息的对象

def parallelize(func: Optional[Callable] = None, *, devices=None, policy="fddps", method="",num_microbatch=1):
    """
    parallelize a function

    Example:
    ```python
    @parallelize
    def compute(x, y):
        out = x + y
        out = x * out
        return out
    ```
    """

    def decorator(func):
        make_ctx = MakeScheduleContext(func, devices or (), policy or "fddps", method or "")#用于构建调度上下文

        @functools.wraps(func)
        def wrapper(*args, **kwargs):#代替被装饰的函数，先将参数以及关键字抽象化，构建上下文调度，然后按照拓扑的顺序，调度每一层中的所有块的执行，生成最终的输出
            if method=="PipeshardParallel":
                virtual_mesh=get_global_virtual_physical_mesh()


                ##
                f = lu.wrap_init(func)
                static_argnums = auto_static_argnums(args)
                if static_argnums:
                    dyn_argnums = [
                        i for i in range(len(args)) if i not in static_argnums
                    ]
                    # Freeze static dict to make it hashable
                    frozen_args = []
                    for i, arg in enumerate(args):
                        if i in static_argnums and isinstance(arg, dict):
                            frozen_args.append(FrozenDict(arg))
                        else:
                            frozen_args.append(arg)
                    f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
                else:
                    dyn_args = args
                
                args_flat, in_tree = tree_flatten(dyn_args)
                f, out_tree = flatten_fun_nokwargs(f, in_tree)
                batch_invars = donation_vector((1,), dyn_args, kwargs)
                abstract_args = map(abstractify_with_aval, args_flat)
                abstract_args_micro = map(abstractify_with_aval, args_flat)
                list_instr={}
                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        abstract_args = map(abstractify_with_aval, args_flat)
                
                        list_instr[mesh_idx]=worker.get_stages_to_run.remote(func,f,parallel_method.policy, parallel_method.method, args, kwargs,batch_invars, parallel_method.num_microbatch, args_flat,parallel_method.layer_num, *abstract_args)
              
                def run_executable(worker,instructions):
                    for instruction in instructions:
                        if instruction.opcode == PipelineInstType.RUN:
                            worker.run_model_parallelism.remote(instruction.stage_id,
                                                    instruction.micro_batch_id,
                                                    instruction.input_vars,
                                                    instruction.output_vars
                                                    )
                        elif instruction.opcode == PipelineInstType.SEND:
                            worker.do_send.remote(instruction.micro_batch_id,
                                        instruction.output_vars,
                                        instruction.dst_rank,
                                        instruction.groupname,
                                        )
                        elif instruction.opcode == PipelineInstType.RECV:
                            worker.do_recv.remote(instruction.micro_batch_id,
                                        instruction.input_vars,
                                        instruction.src_rank,
                                        instruction.groupname,
                                        )
                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        run_executable(worker,list_instr[mesh_idx])

                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        if mesh_idx==0:
                            result=ray.get(worker.return_result.remote())
                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        worker.free_buffers.remote()



            else:
                make_ctx.args, make_ctx.kwargs = args, kwargs
                in_avals, flat_args, _= _abstractify(args, kwargs) 

                pr, out_tree = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)#生成jax语法树
                ctx = make_ctx(pr) #(ShapedArray(int32[], weak_type=True), ShapedArray(int32[], weak_type=True)) 

            return result
        
        
        def run_context(*args, **kwargs):#用于在不执行函数的情况下创建并返回调度上下文
            make_ctx.args, make_ctx.kwargs = args, kwargs
            in_avals, _, _ = _abstractify(args, kwargs)
            ctx = make_ctx(tuple(in_avals))
            return ctx
        
        wrapper.run_context = run_context
        return wrapper #添加属性，提供外界访问


    if func is None:
        return decorator
    return decorator(func)


def compile_executable(
        fun: lu.WrappedFun,
        batch_invars,
        *avals):

    closed_jaxpr_one, micro_batch_size = trace_jaxpr_with_micro_batch(
        fun, batch_invars, 1, avals)
    for store in fun.stores:
        if store:
            store.reset()
    closed_jaxpr_four, micro_batch_size = trace_jaxpr_with_micro_batch(
       fun, batch_invars, 4, avals)

def grad(*args, **kwargs):
    """This is the same as jax.grad, except that alpa inserts a
    gradient marker after the gradient computation.

    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how the gradients are averaged.
    """

    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        print(call_args,call_kwargs)
        arg_list = list(args)
        arg_list[0] = layer_level_transformation(arg_list[0],layer_num=4)
        grad_func = api.grad(*arg_list, **kwargs)
        grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient(grads)

    return ret


def value_and_grad(*args, **kwargs):
    """This is the same as jax.value_and_grad, except that alpa inserts a
    gradient marker after the gradient computation.


    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how the gradients are averaged.
    """
    
    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        arg_list = list(args)
        arg_list[0] = layer_level_transformation(arg_list[0],layer_num=4)
        grad_func = api.value_and_grad(*arg_list, **kwargs)
        val, grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient((val, grads))

    return ret
