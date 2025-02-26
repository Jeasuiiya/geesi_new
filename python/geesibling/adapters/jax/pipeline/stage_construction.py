import itertools
import logging
import numpy as np
from typing import Callable, Optional, Sequence
import jax
from jax import linear_util as lu
from jax._src.lib import xla_client as xc
from jax.core import  AbstractValue, ClosedJaxpr,Literal,Jaxpr,DropVar, gensym
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef

from geesibling.adapters.jax.pipeline.pipeline_schedules import gen_dependency_with_stages, create_pipeline_schedule
from geesibling.adapters.jax.pipeline.apply_grad import (process_apply_gradient, split_compute_grad_and_apply_grad, 
                                                         mark_missing_vars_in_backward_jaxpr_pipeline_marks, apply_grad_get_mean,
                                                         jaxprs_sub_marker,compute_grad_to_accumulate_grad)
from geesibling.adapters.jax.pipeline.primitive_def import pipeline_p
from geesibling.adapters.jax.pipeline.instructions import PipelineInstEmitter

from geesibling.adapters.jax.pipeline.util import trace_jaxpr_with_micro_batch

def compile_executable(
        fun: lu.WrappedFun,
        batch_invars,
        num_microbatch,
        layer_num,
        *avals):
    for store in fun.stores:
        if store:
            store.reset()

    closed_jaxpr, micro_batch_size = trace_jaxpr_with_micro_batch(
            fun, batch_invars, num_microbatch, avals)
    flag=len(closed_jaxpr.jaxpr.invars)

    if num_microbatch > 1:
        for store in fun.stores:
            if store:
                store.reset()
        full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
            fun, batch_invars, 1, avals)

    else:
        full_batch_closed_jaxpr = None


    (jax_all_stages,instructions,global_invars,global_outvars,outvars_map,reduction_vector)= compile_pipeline_executable(closed_jaxpr,full_batch_closed_jaxpr, num_microbatch,batch_invars,layer_num)
    return (jax_all_stages,instructions,global_invars,global_outvars,outvars_map,reduction_vector,flag)


def compile_pipeline_executable(
        closed_jaxpr: ClosedJaxpr,
        full_batch_closed_jaxpr,
        num_microbatch,
        batch_invars,
        layer_num
        ):
    """
    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        global_output_shardings: Forcibly set sharding specs of global
          output vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
    """
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    gensym_func = gensym([closed_jaxpr.jaxpr])

    (closed_jaxpr, jax_pipeline_stages, apply_grad_jaxpr,
     microbatch_bound, global_outvars,acc_grad_jaxpr) = split_and_process_layers(closed_jaxpr,full_batch_closed_jaxpr, num_microbatch, gensym_func)

    stage_to_mesh = [o for o in range(layer_num)]+[o for o in range(layer_num-1,-1,-1)]


    (sliced_apply_grad_stages, apply_grad_placement, global_outvars) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, jax_pipeline_stages, stage_to_mesh,
         gensym_func,  global_outvars)


    dependency = gen_dependency_with_stages(jax_pipeline_stages, sliced_apply_grad_stages)


    jax_pipeline_stages, outvars_map = jaxprs_sub_marker(jax_pipeline_stages)

    jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages

    schedule = create_pipeline_schedule(
        '1f1b',
        dependency=dependency,
        meshes=[o for o in range(1,layer_num+1)],
        apply_grad_placement=apply_grad_placement,
        num_batch=num_microbatch)

    list_acc_var=[]
    for var in acc_grad_jaxpr.jaxpr.invars:
        if  var not in global_invars:
            list_acc_var.append(var)
    emitter_kwargs = dict(jax_all_stages=jax_all_stages,
                            global_invars=global_invars+list_acc_var,
                            global_outvars=global_outvars,
                            mesh_group=[0,1,2,3],
                            schedule=schedule,
                            num_microbatch=num_microbatch,
                            stage_to_mesh=stage_to_mesh+list(apply_grad_placement.values()),
                            outvars_map=outvars_map)
    
    emitter_cls = PipelineInstEmitter
    instruction_lists = emitter_cls(**emitter_kwargs).compile()
    batch_invars=list(batch_invars)
    batch_invars+=[False]*len(list_acc_var)
    return (jax_all_stages,instruction_lists,global_invars+list_acc_var,global_outvars,outvars_map,batch_invars)
def slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr: ClosedJaxpr) -> Sequence[ClosedJaxpr]:
    """Slice a closed jaxpr into multiple JaxPipelineComputation by full
    pipeline markers."""
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_jaxprs = []
    current_jaxpr = None
    current_consts = {}
    current_invars = []
    current_outvars = []
    current_eqns = []

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params["mark_type"] == "start":
            assert current_jaxpr is None, (
                "Defining a pipeline computation "
                "inside a pipeline computation is "
                "not allowed.")
            for var in eqn.invars:
                if isinstance(var, Literal):
                    pass
                elif var in global_consts_dir:
                    current_consts[var] = global_consts_dir[var]
                else:
                    current_invars.append(var)

        for var in eqn.invars:
            if not isinstance(var, Literal) and var in global_consts_dir:
                current_consts[var] = global_consts_dir[var]

        current_eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params["mark_type"] == "end":

            for var in eqn.outvars:
                current_outvars.append(var)
            jaxpr = Jaxpr(
                constvars=list(current_consts.keys()),
                invars=current_invars,
                outvars=current_outvars,
                eqns=current_eqns,
            )
            current_jaxpr = ClosedJaxpr(jaxpr, list(current_consts.values()))
            result_jaxprs.append(current_jaxpr)
            current_jaxpr = None
            current_consts = {}
            current_invars = []
            current_outvars = []
            current_eqns = []

    return result_jaxprs    




# 函数中包累计梯度的生成，对比micro_batch为1和micro_batch大于1的pr添加累加的eqn算子
def split_and_process_layers(closed_jaxpr, full_batch_closed_jaxpr, num_microbatch, gensym_func):

    (closed_jaxpr, compute_grad_jaxpr, apply_grad_jaxpr,
     microbatch_bound) = split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch)
    global_outvars = closed_jaxpr.jaxpr.outvars

#    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(compute_grad_jaxpr)
#    jax_pipeline_layers = (mark_missing_vars_in_backward_jaxpr_pipeline_marks(
#        jax_pipeline_layers, compute_grad_jaxpr.jaxpr.invars, compute_grad_jaxpr.jaxpr.outvars, gensym_func))

    (reduction_vector, post_microbatch_bound,
     _) = _get_full_batch_apply_grad(full_batch_closed_jaxpr, microbatch_bound,
                                     num_microbatch)

    # 对某些变量，在使用微批次时需要进行累加，所以在pr对应的位置添加一个add操作，完成累加，这样就变成了累计梯度的pr
    (acc_grad_jaxpr, microbatch_bound,
     accumulator_mapping) = compute_grad_to_accumulate_grad(
         compute_grad_jaxpr, microbatch_bound, reduction_vector, gensym_func,
         num_microbatch)
    acc_grad_invars = acc_grad_jaxpr.jaxpr.invars
    acc_grad_outvars = acc_grad_jaxpr.jaxpr.outvars
    jax_pipeline_layers = slice_closed_jaxpr_by_full_pipeline_marks(acc_grad_jaxpr)

    # 生成出来的jax_pipeline_layers是JaxPipelineComputation类对象，可以使用self.closed_jaxpr()获取这部分计算的jaxpr
    jax_pipeline_layers = (mark_missing_vars_in_backward_jaxpr_pipeline_marks(
        jax_pipeline_layers, acc_grad_jaxpr.jaxpr.invars, acc_grad_jaxpr.jaxpr.outvars, gensym_func))


    apply_grad_jaxpr, global_outvars = apply_grad_get_mean(
        apply_grad_jaxpr, global_outvars, microbatch_bound.outvars, gensym_func,
        num_microbatch, reduction_vector)
    return (closed_jaxpr, jax_pipeline_layers, apply_grad_jaxpr,
            microbatch_bound, global_outvars,acc_grad_jaxpr)




def _get_full_batch_apply_grad(closed_jaxpr,
                               microbatch_bound,
                               num_microbatch,
                               batch_dim=0):
    """
    Compare the micro-batch jaxpr and full-batch jaxpr. Return whether
    the out var's is reduced across micro-batches.

    TODO(yonghao): the reduction vector should be created by a
    more careful analysis.
    """
    if num_microbatch == 1:
        reduced_vector = [True] * len(microbatch_bound.outvars)
        post_microbatch_bound = microbatch_bound
        apply_grad_jaxpr = None
        return reduced_vector, post_microbatch_bound, apply_grad_jaxpr

    gensym_func = gensym([closed_jaxpr.jaxpr])
    (_, _, apply_grad_jaxpr,
     post_microbatch_bound) = (split_compute_grad_and_apply_grad(
         closed_jaxpr, gensym_func, num_microbatch))
    reduced_vector = []
    for mb_var, var in zip(microbatch_bound.outvars,
                           post_microbatch_bound.outvars):
        microbatch_shape = mb_var.aval.shape
        batch_shape = var.aval.shape
        if microbatch_shape != batch_shape:
            expected_microbatched_shape = list(batch_shape)
            assert expected_microbatched_shape[batch_dim] % num_microbatch == 0
            expected_microbatched_shape[batch_dim] //= num_microbatch
            assert tuple(expected_microbatched_shape) == microbatch_shape
            if len(apply_grad_jaxpr.eqns) > 0:
                raise NotImplementedError(
                    "Some vars marked by gradient markers are not reduced "
                    "but concatenated. This case in the training mode "
                    "is not supported yet.")
        reduced_vector.append(microbatch_shape == batch_shape)

    return reduced_vector, post_microbatch_bound, apply_grad_jaxpr



