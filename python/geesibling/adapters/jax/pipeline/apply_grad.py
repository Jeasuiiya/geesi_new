import numpy as np
from typing import Sequence, Dict, Tuple
from jax.core import (ClosedJaxpr, Var, Jaxpr, DropVar, Literal, get_aval, raise_to_shaped,JaxprEqn)

from jax.lax import  div_p
from jax._src.util import safe_map
from jax.lax import add_p, div_p, and_p, or_p
from geesibling.adapters.jax.pipeline.util import (OrderedSet, get_var_mapping, 
                                                    clone_jaxpr_eqn, slices_to_jaxpr, new_jaxpr_eqn, clone_jaxpr)
from geesibling.adapters.jax.pipeline.primitive_def import pipeline_p, mark_pipeline_jaxpreqn
from geesibling.adapters.jax.pipeline.util import clone_jaxpr,clone_jaxpr_eqn
unsafe_map, map = map, safe_map  # type: ignore


def split_compute_grad_and_apply_grad(closed_jaxpr: ClosedJaxpr, gensym_fn,
                                      num_microbatch: int):
    """Split the train_step jaxpr into two parts: compute_grad and
    apply_grad. These two parts are separated by a gradient marker generated
    by `alpa.grad`."""
    # Locate the marker
    split_eqn = None
    for idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'grad':
            split_eqn = eqn
            split_idx = idx
    
    sliced_eqns = [
        closed_jaxpr.eqns[:split_idx], [split_eqn],
        closed_jaxpr.eqns[split_idx + 1:]
    ]

    # 重写jaxpr，把计算图中跨层的计算式加到应用梯度计算式的开头部分，
    closed_jaxpr = _rewrite_cross_layer_grad(*sliced_eqns, gensym_fn, closed_jaxpr)

    closed_jaxpr, sliced_eqns = _remove_replicated_marked_var(closed_jaxpr)

    # Reconstruct jaxpr
    sliced_jaxprs = slices_to_jaxpr(closed_jaxpr, sliced_eqns)
    compute_grad, _, apply_grad = sliced_jaxprs  # pylint: disable=unbalanced-tuple-unpacking
    split_eqn = sliced_eqns[1][0]
    
    assert len(split_eqn.invars) == len(split_eqn.outvars)
    invars_without_dropvar = []
    outvars_without_dropvar = []
    for invar, outvar in zip(split_eqn.invars, split_eqn.outvars):
        if not isinstance(outvar, DropVar):
            invars_without_dropvar.append(invar)
            outvars_without_dropvar.append(outvar)
    split_eqn = clone_jaxpr_eqn(split_eqn, invars_without_dropvar,
                                outvars_without_dropvar)
    return closed_jaxpr, compute_grad, apply_grad, split_eqn



def mark_missing_vars_in_backward_jaxpr_pipeline_marks(
        sliced_jaxprs: Sequence[Jaxpr], global_invars,
        global_outvars, gensym_func):

    assert len(sliced_jaxprs) % 2 == 0.
    num_forward_prs = len(sliced_jaxprs) // 2
    var_pr_id = {}
    for var in global_invars:
        if not isinstance(var, Literal):
            var_pr_id[var] = -1

    pr_marked_to_unmarked_invars = [{} for _ in sliced_jaxprs]
    pr_weight_invars = [{} for _ in sliced_jaxprs]
    pr_additional_invars = [OrderedSet() for _ in sliced_jaxprs]
    pr_additional_outvars = [OrderedSet() for _ in sliced_jaxprs]
    for pr_id, pr in enumerate(sliced_jaxprs):
        current_consts = dict(
            zip(pr.jaxpr.constvars, pr.consts))
        pr = pr.jaxpr
        for eqn in pr.eqns:
            if eqn.primitive == pipeline_p and eqn.params[
                    "mark_type"] == "start":
                for invar, outvar in zip(eqn.invars, eqn.outvars):
                    pr_marked_to_unmarked_invars[pr_id][
                        outvar] = invar
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in current_consts and
                        var not in pr.invars):
                    source_pr_id = var_pr_id[var]
                    if source_pr_id != pr_id:
                        # Special case for the model weights. If a backward
                        # pr is using an invar of a forward
                        # pr, do not let the invar go into the stage.
                        # Instead, we can directly use the original invar.
                        if (pr_id >= num_forward_prs and
                                source_pr_id
                                == 2 * num_forward_prs -
                                pr_id - 1 and
                                var in pr_marked_to_unmarked_invars[
                                    source_pr_id]):
                            pr_weight_invars[pr_id][var] = (
                                pr_marked_to_unmarked_invars[
                                    source_pr_id][var])
                            continue
                        # Mark all the variables in the backward pr
                        # that are not currently defined in pipeline markers.
                        if (source_pr_id != -1 and var not in
                                sliced_jaxprs[source_pr_id].jaxpr.outvars):
                            pr_additional_outvars[
                                source_pr_id].add(var)
                        pr_additional_invars[pr_id].add(var)
            for var in eqn.outvars:
                var_pr_id[var] = pr_id

    for var in global_outvars:
        source_pr_id = var_pr_id[var]
        if source_pr_id != -1 and var not in sliced_jaxprs[
                source_pr_id].jaxpr.outvars:
            pr_additional_outvars[source_pr_id].add(var)

    result_jaxprs = []
    for i, pr in enumerate(sliced_jaxprs):
        current_eqns = []
        current_consts = dict(
            zip(pr.jaxpr.constvars, pr.consts))
        pr = pr.jaxpr
        assert (pr.eqns[0].primitive is pipeline_p and
                pr.eqns[0].params["mark_type"] == "start")
        assert (pr.eqns[-1].primitive is pipeline_p and
                pr.eqns[-1].params["mark_type"] == "end")

        pr_var_mapping = {
            var: gensym_func(var.aval)
            for var in pr_additional_invars[i] |
            pr_additional_outvars[i] |
            pr_weight_invars[i].keys()
        }
        pipeline_start_invars = list(pr.eqns[0].invars)
        pipeline_start_outvars = [
            get_var_mapping(pr_var_mapping, var)
            for var in pr.eqns[0].outvars
        ]
        for var in pr_additional_invars[i]:
            pipeline_start_invars.append(var)
            pipeline_start_outvars.append(pr_var_mapping[var])
        for marked_var, unmarked_var in pr_weight_invars[i].items():
            pipeline_start_invars.append(unmarked_var)
            pipeline_start_outvars.append(pr_var_mapping[marked_var])

        pipeline_start_invars_without_literal = []
        pipeline_start_outvars_without_literal = []
        for invar, outvar in zip(pipeline_start_invars, pipeline_start_outvars):
            if isinstance(invar, Literal):
                pr_var_mapping[outvar] = invar
            else:
                pipeline_start_invars_without_literal.append(invar)
                pipeline_start_outvars_without_literal.append(outvar)

        current_invars = list(pipeline_start_invars_without_literal)
        current_eqns.append(pr.eqns[0]._replace(
            invars=pipeline_start_invars_without_literal,
            outvars=pipeline_start_outvars_without_literal))

        for eqn in pr.eqns[1:-1]:
            invars = [
                get_var_mapping(pr_var_mapping, var)
                for var in eqn.invars
            ]
            outvars = [
                get_var_mapping(pr_var_mapping, var)
                for var in eqn.outvars
            ]
            current_eqns.append(
                eqn._replace(invars=invars, outvars=outvars))
        
        pipeline_end_invars = [
            get_var_mapping(pr_var_mapping, var)
            for var in pr.eqns[-1].invars
        ]
        pipeline_end_outvars = list(pr.eqns[-1].outvars)
        for var in pr_additional_outvars[i]:
            pipeline_end_invars.append(pr_var_mapping[var])
            pipeline_end_outvars.append(var)

        pipeline_end_invars_without_dropvar = []
        pipeline_end_outvars_without_dropvar = []
        for invar, outvar in zip(pipeline_end_invars, pipeline_end_outvars):
            if not isinstance(outvar, DropVar):
                pipeline_end_invars_without_dropvar.append(invar)
                pipeline_end_outvars_without_dropvar.append(outvar)

        current_outvars = list(pipeline_end_outvars_without_dropvar)

        current_eqns.append(pr.eqns[-1]._replace(
            invars=pipeline_end_invars_without_dropvar,
            outvars=pipeline_end_outvars_without_dropvar))
        
        jaxpr = Jaxpr(
                constvars=list(current_consts.keys()),
                invars=current_invars,
                outvars=current_outvars,
                eqns=current_eqns,
            )
        current_jaxpr = ClosedJaxpr(jaxpr, list(current_consts.values()))
        result_jaxprs.append(current_jaxpr)

    return result_jaxprs



def get_var_to_mesh(invars: Sequence[Var],
                    computation_jaxprs: Sequence[Jaxpr],
                    stage_to_mesh: Dict[int, int], apply_in_to_acc_out):
    """Get the mapping from variables to mesh."""
    # TODO(yonghao): now assume all gradients are variables(not literal)
    outvar2mesh = {}
    for i, computation_jaxpr in enumerate(computation_jaxprs):
        for var in computation_jaxpr.jaxpr.outvars:
            if isinstance(var, Var):
                outvar2mesh[var] = stage_to_mesh[i]
    return {
        invar: outvar2mesh[apply_in_to_acc_out[invar]]
        for invar in invars
        if ((invar in apply_in_to_acc_out) and
            (apply_in_to_acc_out[invar] in outvar2mesh))
    }



def process_apply_gradient(apply_grad_jaxpr, microbatch_bound, pipeline_stages,
                           stage_to_mesh, gensym_func, global_outvars):
    """Slice apply_grad jaxpr into stages and assign them to the corresponding
    meshes.
    1、根据grad边界的输出判断生成的每一个梯度属于哪一个stage,来判断应用梯度中使用梯度的eqn属于哪一个stage,进行划分
    2、除此外，还要对应用梯度中使用到的全局输入进行判断。属于哪一个stage，并把其eqn也划分到stage中
    可能存在的问题是，一个全局变量可能会在多个stage中使用，那么整个全局变量应该划分到哪一个stage就是个问题了
    （可能的解决办法是，重构apply_jaxpr的输入，对多个stage使用的变量进行多次输入）
    """
    # Process apply gradient:
    # change invars of apply grad to outvars of accumulate grad
    eqn_num = len(apply_grad_jaxpr.eqns)
    gradients = microbatch_bound.outvars
    apply_in_to_acc_out = dict(zip(gradients, microbatch_bound.invars))
    # 根据梯度变量和流水线阶段的映射关系，确定每个梯度变量所在的网格
    gradvar_to_mesh = get_var_to_mesh(gradients, pipeline_stages, stage_to_mesh,
                                      apply_in_to_acc_out)
    #print("gradvar_to_mesh:",gradvar_to_mesh)
    #gradvar_to_mesh = sorted(gradvar_to_mesh.items(), key=lambda x: x[1], reverse=False)
    var_stage_map = {}
    eqn_stage_map = {}
    for i, eqn in enumerate(apply_grad_jaxpr.eqns):
        for var in eqn.invars:
            if not isinstance(var,Literal):
                if var in gradvar_to_mesh:
                    if i in eqn_stage_map:
                        eqn_stage_map[i] = min(gradvar_to_mesh[var],eqn_stage_map[i])
                    else:
                        eqn_stage_map[i] = gradvar_to_mesh[var]
    for i , eqn in enumerate(apply_grad_jaxpr.eqns):
        if i in eqn_stage_map:
            for var in eqn.invars:
                if not isinstance(var,Literal):
                    var_stage_map[var] = eqn_stage_map[i]
            for var in eqn.outvars:
                var_stage_map[var] = eqn_stage_map[i]
    for i , eqn in enumerate(apply_grad_jaxpr.eqns):
        for var in eqn.invars:
            if not isinstance(var,Literal):
                if var in var_stage_map:
                    if i not in eqn_stage_map:
                        eqn_stage_map[i] = var_stage_map[var]
                    else:
                        eqn_stage_map[i] = min(var_stage_map[var],eqn_stage_map[i])
                    for inv in eqn.invars:
                        if not isinstance(inv,Literal):
                            var_stage_map[inv] = eqn_stage_map[i]
                    for outv in eqn.outvars:
                        var_stage_map[outv] = eqn_stage_map[i]
                    
    for i, eqn in enumerate(apply_grad_jaxpr.eqns[::-1]):
        for var in eqn.outvars:
            if var in var_stage_map:
                if (eqn_num - i - 1) not in eqn_stage_map:
                    eqn_stage_map[eqn_num - i - 1] = var_stage_map[var]
                else:
                    eqn_stage_map[eqn_num - i - 1] = min(var_stage_map[var],eqn_stage_map[eqn_num - i - 1])
                for inv in eqn.invars:
                        if not isinstance(inv,Literal):
                            var_stage_map[inv] = eqn_stage_map[eqn_num - i - 1]
    #print(var_stage_map)
    #print(eqn_stage_map)
    
    apply_stage_size = len(pipeline_stages) // 2

    sliced_eqns = [[] for _ in range(apply_stage_size)]

    for i, eqn in enumerate(apply_grad_jaxpr.eqns):
        if i in eqn_stage_map:
            sliced_eqns[eqn_stage_map[i]].append(eqn)
        else:
            sliced_eqns[apply_stage_size - 1].append(eqn)
    


    sliced_apply_grad_stages = slices_to_jaxpr(apply_grad_jaxpr, sliced_eqns)
    
    apply_grad_placement = {}
    for i in range(apply_stage_size):
        apply_grad_placement[len(pipeline_stages) + i] = stage_to_mesh[i]


    sliced_apply = []
    for i in sliced_apply_grad_stages:
        new_invars = [
            get_var_mapping(apply_in_to_acc_out, var)
            for var in i.jaxpr.invars
        ]
        new_eqns = []
        for eqn in i.eqns:
            new_eqn_invars = [
                get_var_mapping(apply_in_to_acc_out, var)
                for var in eqn.invars
            ]
            new_eqn = new_jaxpr_eqn(new_eqn_invars, eqn.outvars, eqn.primitive, eqn.params)
            new_eqns.append(new_eqn)

        sliced_apply.append(clone_jaxpr(i, new_invars, i.jaxpr.outvars, new_eqns))

    return (sliced_apply, apply_grad_placement, global_outvars)


def _value_to_literal(value, dtype):
    literal_val = np.array(value, dtype)
    return Literal(literal_val, raise_to_shaped(get_aval(literal_val)))

def replace_all_with(closed_jaxpr: ClosedJaxpr, mapping):
    """Replace all variables in a jaxpr given the mapping."""

    def map_var(var):
        return get_var_mapping(mapping, var)

    new_glob_invars = [map_var(var) for var in closed_jaxpr.jaxpr.invars]
    new_glob_outvars = [map_var(var) for var in closed_jaxpr.jaxpr.outvars]
    new_eqns = []
    for eqn in closed_jaxpr.eqns:
        new_invars = [map_var(var) for var in eqn.invars]
        new_outvars = [map_var(var) for var in eqn.outvars]
        new_eqns.append(clone_jaxpr_eqn(eqn, new_invars, new_outvars))
    new_jaxpr = clone_jaxpr(closed_jaxpr, new_glob_invars, new_glob_outvars,
                            new_eqns)
    return new_jaxpr


def apply_grad_get_mean(apply_grad_jaxpr, global_outvars, gradients, gensym_fn,
                        num_microbatch, reduce_invars):
    """
    Get the mean of input (accumulated) gradients and run apply gradient.

    If the input is output, after this transform it outputs the divided version.
    """
    mapping = {}
    new_eqns = []
    invar_set = OrderedSet(apply_grad_jaxpr.jaxpr.invars)
    outvar_set = OrderedSet(apply_grad_jaxpr.jaxpr.outvars)
    for invar, reduce in zip(gradients, reduce_invars):
        if not reduce:
            mapping[invar] = invar
            continue
        div_out = gensym_fn(invar.aval)
        new_eqns.append(
            new_jaxpr_eqn([
                invar,
                _value_to_literal(num_microbatch, invar.aval.dtype),
            ], [div_out], div_p, {}))
        mapping[invar] = div_out
    replaced = replace_all_with(apply_grad_jaxpr, mapping)
    final_invars = list(apply_grad_jaxpr.jaxpr.invars)
    final_outvars = list(replaced.jaxpr.outvars)
    for invar, reduce in zip(gradients, reduce_invars):
        if not reduce:
            continue
        if invar not in invar_set:
            final_invars.append(invar)
        if invar in global_outvars and invar not in outvar_set:
            # use the divided version to replace the original one
            final_outvars.append(mapping[invar])
    new_eqns.extend(replaced.jaxpr.eqns)
    new_jaxpr = clone_jaxpr(apply_grad_jaxpr, final_invars, final_outvars,
                            new_eqns)
    global_outvars = [get_var_mapping(mapping, var) for var in global_outvars]
    return new_jaxpr, global_outvars

def apply_grad_add_marker(jaxprs: Sequence[ClosedJaxpr],
                          apply_in_to_acc_out: Dict[Var, Var],
                          gensym_fn):
    """Add pipeline markers for sliced apply grads, keep invars and outvars
    still unless.

    The invar is in apply_in_to_acc_out or invar is outvar:
    In the first case, the final invar follows the apply_in_to_acc_out;
    In the second case, the final outvar is recorded in outvar_map.

    Args:
        jaxprs: sliced apply grads.
        apply_in_to_acc_out: which output of accumulate grad corresponds to the
            invar of apply grad
        gensym_fn: gensym function of the whole jaxpr.
        computation: output JaxPipelineComputation or ClosedJaxpr.
    """
    results = []
    outvar_map = {}
    for i, jaxpr in enumerate(jaxprs):
        new_map = {}
        for invar in jaxpr.jaxpr.invars:
            if invar not in apply_in_to_acc_out:
                new_map[invar] = gensym_fn(invar.aval)
        for outvar in jaxpr.jaxpr.outvars:
            if not isinstance(outvar, Var):
                raise NotImplementedError(
                    'outvar of apply grad cannot be literal')
            if outvar in jaxpr.jaxpr.invars:
                if outvar not in outvar_map:
                    outvar_map[outvar] = gensym_fn(outvar.aval)
                continue
            new_map[outvar] = gensym_fn(outvar.aval)
        replaced = replace_all_with(jaxpr, new_map).jaxpr
        
        new_invars = [
            get_var_mapping(apply_in_to_acc_out, var)
            for var in jaxpr.jaxpr.invars
        ]
        new_outvars = [
            get_var_mapping(outvar_map, var) for var in jaxpr.jaxpr.outvars
        ]
        APPLY_GRAD_MARKER_SUFFIX = 'apply_grad'
        name = f'{i}_{APPLY_GRAD_MARKER_SUFFIX}'
        start_marker = mark_pipeline_jaxpreqn(new_invars,
                                              replaced.invars,
                                              name=name,
                                              mark_type='start')
        end_marker = mark_pipeline_jaxpreqn(replaced.outvars,
                                            new_outvars,
                                            name=name,
                                            mark_type='end')
        new_eqns = [start_marker] + replaced.eqns + [end_marker]
        
        new_jaxpr = Jaxpr(
            constvars=list(jaxpr.jaxpr.constvars),
            invars=new_invars,
            outvars=new_outvars,
            eqns=new_eqns,
        )
        current_jaxpr = ClosedJaxpr(new_jaxpr, list(jaxpr.consts))
        results.append(current_jaxpr)

    outvar_map.update(apply_in_to_acc_out)
    return results, outvar_map


def jaxprs_sub_marker(jaxprs: Sequence[ClosedJaxpr]):
    results = []
    outvars_map = {}
    for i,jaxpr in enumerate(jaxprs):

        pipeline_start_map = {}
        for invar, outvar in zip(jaxpr.eqns[0].invars,jaxpr.eqns[0].outvars):
            pipeline_start_map[outvar] = invar
        
        for invar, outvar in zip(jaxpr.eqns[-1].invars,jaxpr.eqns[-1].outvars):
            #outvar_map[invar] = outvar
            outvars_map[outvar] = invar
        new_invars = jaxpr.jaxpr.invars
        new_outvars = jaxpr.eqns[-1].invars
        
        new_eqns = []
        for eqn in jaxpr.eqns[1:len(jaxpr.eqns)-1]:
            new_eqn_invars = []
            for var in eqn.invars:
                if not isinstance(var, Literal):
                    if var not in pipeline_start_map:
                        new_eqn_invars.append(var)
                    else:
                        new_eqn_invars.append(pipeline_start_map[var])
                else:
                    new_eqn_invars.append(var)
            #for var in eqn.outvars:
            #    if var in jaxpr.eqns[-1].invars:
            #        new_outvars.append(var)
            
            new_eqn = new_jaxpr_eqn(new_eqn_invars, eqn.outvars, eqn.primitive, eqn.params)
            new_eqns.append(new_eqn)
        

        new_jaxpr = Jaxpr(
            constvars=list(jaxpr.jaxpr.constvars),
            invars=new_invars,
            outvars=new_outvars,
            eqns=new_eqns,
        )
        current_jaxpr = ClosedJaxpr(new_jaxpr, list(jaxpr.consts))
        results.append(current_jaxpr)
    return results, outvars_map

def compute_grad_to_accumulate_grad(
        compute_jaxpr: ClosedJaxpr, microbatch_bound: JaxprEqn,
        reduction_vector: Sequence[bool], gensym_fn,
        num_microbatch) -> Tuple[ClosedJaxpr, JaxprEqn, Dict[Var, Var]]:

    if num_microbatch <= 1:
        return compute_jaxpr, microbatch_bound, {}
    post_to_pre_marker_outs = _get_post_to_pre_marker_mapping(compute_jaxpr)
    to_reduce_pre_marker_outs = []
    for var, reduced in zip(compute_jaxpr.jaxpr.outvars, reduction_vector):
        if reduced:
            to_reduce_pre_marker_outs.append(post_to_pre_marker_outs[var])
    reduced_invars = {
        outvar: gensym_fn(outvar.aval) for outvar in to_reduce_pre_marker_outs
    }
    reduced_outvars = {
        outvar: gensym_fn(outvar.aval) for outvar in to_reduce_pre_marker_outs
    }

    new_glob_outvars = []
    new_glob_invars = compute_jaxpr.jaxpr.invars + []

    update_outs = {}
    reduced_in_to_out = {}
    for outvar, reduced in zip(compute_jaxpr.jaxpr.outvars, reduction_vector):
        if not reduced:
            new_glob_outvars.append(outvar)
            update_outs[outvar] = outvar
        elif isinstance(outvar, Var):
            assert outvar in post_to_pre_marker_outs
            pre_marker_outvar = post_to_pre_marker_outs[outvar]
            reduced_outvar = reduced_outvars[pre_marker_outvar]
            reduced_invar = reduced_invars[pre_marker_outvar]

            new_glob_outvars.append(reduced_outvar)
            new_glob_invars.append(reduced_invar)
            update_outs[outvar] = reduced_outvar
            reduced_in_to_out[reduced_invar] = reduced_outvar
        else:
            raise NotImplementedError('outputs cannot be Literal')

    new_eqns = _rewrite_jaxpr_to_reduced_outputs(compute_jaxpr,
                                                 to_reduce_pre_marker_outs,
                                                 reduced_invars,
                                                 reduced_outvars, gensym_fn)



    new_closed_jaxpr = clone_jaxpr(compute_jaxpr, new_glob_invars,
                                   new_glob_outvars, new_eqns)
     
    microbatch_bound_invars = [update_outs[x] for x in microbatch_bound.invars]
    microbatch_bound = clone_jaxpr_eqn(microbatch_bound,
                                       microbatch_bound_invars)
    

                 
    return new_closed_jaxpr, microbatch_bound, reduced_in_to_out


def _get_post_to_pre_marker_mapping(compute_jaxpr):
    """
    Get a dict that maps an out_var of a pipeline marker to
    its corresponding in_var.
    """
    post_marker_outs = _filter_droped(compute_jaxpr.jaxpr.outvars)
    # Currently, assume no grad is literal
    assert len(post_marker_outs) == len(compute_jaxpr.jaxpr.outvars)
    post_marker_outs = OrderedSet(post_marker_outs)
    # from post_marker_outs to post_to_pre_marker_outs(cross pipeline marker)
    post_to_pre_marker_outs = {}
    pre_to_post_marker_outs = {}
    for eqn in reversed(compute_jaxpr.eqns):
        if eqn.primitive is pipeline_p:
            for i, outvar in enumerate(eqn.outvars):
                if outvar in post_marker_outs:
                    post_to_pre_marker_outs[outvar] = eqn.invars[i]
                    pre_to_post_marker_outs[eqn.invars[i]] = outvar
                elif outvar in pre_to_post_marker_outs:
                    # in case that:
                    #   invar = compute gradient
                    #   invar' = pipeline end(invar)
                    #   outvar = pipeline start(invar')
                    #   final = pipeline end(outvar)
                    # post_to_pre_marker_outs[final] = invar' instead of outvar
                    final_outvar = pre_to_post_marker_outs[outvar]
                    post_to_pre_marker_outs[final_outvar] = eqn.invars[i]
                    pre_to_post_marker_outs[eqn.invars[i]] = final_outvar

    for outvar in post_marker_outs:
        assert outvar in post_to_pre_marker_outs, (
            'all outputs should be captured by pipeline marker ')
    return post_to_pre_marker_outs


def _filter_droped(vars):
    return [v for v in vars if not isinstance(v, DropVar)]


def _rewrite_jaxpr_to_reduced_outputs(compute_jaxpr, to_reduce_pre_marker_outs,
                                      reduce_invars, reduce_outvars, gensym_fn):
    new_eqns = []
    pipe_start = None
    pipe_eqns = []
    to_acc = []
    to_reduce_pre_marker_outs = OrderedSet(to_reduce_pre_marker_outs)
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'start':
                pipe_start = eqn
                for outvar in eqn.outvars:
                    if (not isinstance(outvar, DropVar) and
                            outvar in to_reduce_pre_marker_outs):
                        # collect to_reduce_pre_marker_outs in this computation
                        to_acc.append(outvar)
                continue
            if eqn.params['mark_type'] == 'end':
                # add grad used in this computation in pipeline start
                reduce_invar_post_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                reduce_outvar_pre_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                new_pipe_start = mark_pipeline_jaxpreqn(
                    pipe_start.invars + map(lambda x: reduce_invars[x], to_acc),
                    pipe_start.outvars +
                    # pylint: disable=cell-var-from-loop
                    map(lambda x: reduce_invar_post_pipe[x], to_acc),
                    pipe_start.params['name'],
                    pipe_start.params['mark_type'])
                new_eqns.append(new_pipe_start)
                # add normal eqns
                new_eqns.extend(pipe_eqns)
                # add acc grad(adds)
                for gradient in to_acc:
                    new_eqns.append(
                        new_jaxpr_eqn(
                            [reduce_invar_post_pipe[gradient], gradient],
                            [reduce_outvar_pre_pipe[gradient]], add_p, {}))
                # add grad created in this computation in pipeline end
                new_pipe_end = mark_pipeline_jaxpreqn(
                    # pylint: disable=cell-var-from-loop
                    eqn.invars +
                    map(lambda x: reduce_outvar_pre_pipe[x], to_acc),
                    eqn.outvars + map(lambda x: reduce_outvars[x], to_acc),
                    eqn.params['name'],
                    eqn.params['mark_type'])
                new_eqns.append(new_pipe_end)
                pipe_start = None
                pipe_eqns = []
                to_acc = []
                continue
        pipe_eqns.append(eqn)
        for outvar in eqn.outvars:
            if (not isinstance(outvar, DropVar) and
                    outvar in to_reduce_pre_marker_outs):
                # collect to_reduce_pre_marker_outs in this computation
                to_acc.append(outvar)
    return new_eqns


def _remove_replicated_marked_var(closed_jaxpr: ClosedJaxpr):
    """Some variables are marked multiple times with the same marker.
    This pass removes them.
    """
    new_eqns = []
    var_map = {}
    mb_idx = None
    for eqn in closed_jaxpr.eqns:
        if eqn.primitive == pipeline_p:
            eqn_map = {}
            new_invars = []
            new_outvars = []
            if eqn.params['mark_type'] == 'grad':
                mb_idx = len(new_eqns)
            #记录每个变量的映射关系并处理多次标记的情况
            for inv, outv in zip(eqn.invars, eqn.outvars):
                if isinstance(outv, DropVar):
                    continue
                if isinstance(inv, Var):
                    if inv in var_map:
                        var_map[outv] = var_map[inv]
                        continue
                    elif inv in eqn_map:
                        var_map[outv] = eqn_map[inv]
                        continue
                if isinstance(inv, Var):
                    eqn_map[inv] = outv
                new_invars.append(inv)
                new_outvars.append(outv)
            new_eqns.append(clone_jaxpr_eqn(eqn, new_invars, new_outvars))
            continue
        new_invars = [get_var_mapping(var_map, v) for v in eqn.invars]
        new_eqns.append(clone_jaxpr_eqn(eqn, new_invars))
    sliced_eqns = new_eqns[:mb_idx], [new_eqns[mb_idx]], new_eqns[mb_idx + 1:]
    new_outvars = [
        get_var_mapping(var_map, v) for v in closed_jaxpr.jaxpr.outvars
    ]
    return clone_jaxpr(closed_jaxpr, outvars=new_outvars,
                       eqns=new_eqns), sliced_eqns

def _rewrite_cross_layer_grad(compute_eqns, microbatch_bound, apply_eqns,
                              gensym_fn, closed_jaxpr):
    """
    (*sliced_eqns, gensym_fn, closed_jaxpr)
    If a parameter is used in multiple stages, its gradient is computed in
    multiple stages and then added together. We accumulate the results on each
    stage, and add them together exactly at the start of apply grad period.

    A common use case is the tied embedding in language models.
    """
    layer_invars, pipeline_outvars = _pipeline_marker_analysis(compute_eqns)
    # {d, b, c, e, h, cr, a, g, y, f} {y: 0, cr: 2, cs: 2, ct: 2, cu: 2, cv: 2, do: 3, dp: 3, dq: 3, dr: 3}
    # Those eqn directly use output of pipeline end is delayed to apply grad.
    cross_layer_grad_eqns, new_compute_eqns = _get_delayed_eqns(
        compute_eqns, layer_invars, pipeline_outvars, gensym_fn)
    # Rewrite microbatch_bound and cross_layer_grad eqns.
    #有修改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    (new_microbatch_bound,
     microbatch_bound_in_to_outs) = _rewrite_microbatch_bound(
         microbatch_bound[0], cross_layer_grad_eqns, gensym_fn)
    # rewrite cross layer grad eqns and insert them to the top of apply eqns.
    new_apply_eqns = _rewrite_delayed_gradient_sum_eqns(
        cross_layer_grad_eqns, microbatch_bound_in_to_outs)
    new_apply_eqns += apply_eqns
    new_global_outvars = list(closed_jaxpr.jaxpr.outvars)
    for idx in range(len(new_global_outvars)):
        var = new_global_outvars[idx]
        if isinstance(var, Literal):
            continue
        if isinstance(var, Var) and var in microbatch_bound_in_to_outs:
            new_global_outvars[idx] = microbatch_bound_in_to_outs[var]
    closed_jaxpr = clone_jaxpr(closed_jaxpr,
                               eqns=new_compute_eqns + [new_microbatch_bound] +
                               new_apply_eqns,
                               outvars=new_global_outvars)
    return closed_jaxpr

def _filter_literal(vars):
    return [v for v in vars if isinstance(v, Var)]


def _filter_droped(vars):
    return [v for v in vars if not isinstance(v, DropVar)]

def _pipeline_marker_analysis(compute_eqns):
    """Get vars as inputs and outputs of layers"""
    layer_invars = set()
    pipeline_outvars = {}
    marker_cnt = 0
    for eqn in compute_eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['mark_type'] == 'end':
                for v in _filter_droped(eqn.outvars):
                    pipeline_outvars[v] = marker_cnt
                marker_cnt += 1
            elif eqn.params['mark_type'] == 'start':
                layer_invars.update(_filter_literal(eqn.invars))
    return layer_invars, pipeline_outvars

def _get_delayed_eqns(compute_eqns, layer_invars, pipeline_outvars, gensym_fn):
    """
    获取可以延迟到应用梯度的方程，并重写不能延迟的方程，将他们移动到同一层中
    Get eqns that can be delayed to apply gradient stage and rewrite eqns that
    cannot do so by moving them into a layer.

    An example of cannot delayed vars is: x is computed in layer0, and sent to
    layer1 and layer2. There is grad(x) = grad_1(x) + grad_2(x), but the
    grad(weight) depends on grad(x) and is in the acc_grad period, so we cannot
    delay it to the apply_grad period.
    """
    cross_layer_grad_eqns = []
    new_compute_eqns = []
    moved_to_layer_eqns = []

    marked_vars = set()
    used_vars = set()
    out_marker = True
    for eqn in reversed(compute_eqns):
        invars = _filter_literal(eqn.invars)
        outvars = _filter_droped(eqn.outvars)
        used_outvars = used_vars.intersection(outvars)
        if eqn.primitive is pipeline_p:
            # invars of a pipeline end marker is marked
            if eqn.params['mark_type'] == 'end':
                marked_vars.update(invars)
                out_marker = False
            else:
                out_marker = True
            new_compute_eqns.append(eqn)
        else:
            # we don't want to do dce here, because it may make its operand be
            # considered as cross layer grad, and then moved across microbatch
            # boundary, which is harder to analyze.
            if len(outvars) == 0 and out_marker:
                continue
            # only if an eqn is not used and is out marker will be it moved
            # after microbatch boundary. Those inside a microbatch boundary is
            # handled by later DCE.
            elif not used_outvars and out_marker:
                cross_layer_grad_eqns.append(eqn)
                continue
            elif marked_vars.issuperset(used_outvars):
                # eqn is marked if all outvars are marked, then mark its invars.
                marked_vars.update(invars)
                new_compute_eqns.append(eqn)
            else:
                assert not marked_vars.intersection(
                    outvars), f"'{eqn}' is partially marked."
                if layer_invars.intersection(outvars):
                    # move the marked var to the latest stage producing some of
                    # its invars.
                    moved_to_layer_eqns.append(eqn)
                    # update layer invars and marked vars.
                    layer_invars.update(invars)
                    marked_vars.update(outvars)
                else:
                    cross_layer_grad_eqns.append(eqn)
                    continue
        used_vars.update(invars)

    new_compute_eqns = list(reversed(new_compute_eqns))
    cross_layer_grad_eqns = list(reversed(cross_layer_grad_eqns))
    eqn_moved_to = {}
    for eqn in reversed(moved_to_layer_eqns):
        invars = _filter_literal(eqn.invars)
        outvars = _filter_droped(eqn.outvars)
        moved_to = max(pipeline_outvars[v] for v in invars)
        eqn_moved_to.setdefault(moved_to, []).append(eqn)
        pipeline_outvars.update({v: moved_to for v in outvars})
    if eqn_moved_to:
        new_compute_eqns = _rewrite_compute_eqns(new_compute_eqns, eqn_moved_to,
                                                 gensym_fn)
    return cross_layer_grad_eqns, new_compute_eqns


def _rewrite_microbatch_bound(microbatch_bound, delayed_eqns, gensym_fn):
    """
    Rewrite the microbatch bound because some eqns are moved from microbatched
    part of the graph to non-microbatched part.
    """
    microbatch_bound_in_to_outs = {}
    for invar, outvar in zip(microbatch_bound.invars, microbatch_bound.outvars):
        if isinstance(invar, Var) and not isinstance(outvar, DropVar):
            microbatch_bound_in_to_outs[invar] = outvar
    delayed_invars = OrderedSet()
    delayed_outvars = OrderedSet()
    for eqn in delayed_eqns:
        delayed_invars.update(_filter_literal(eqn.invars))
        delayed_outvars.update(_filter_droped(eqn.outvars))
    delayed_invars.difference_update(delayed_outvars)
    delayed_invars.difference_update(microbatch_bound_in_to_outs.keys())
    delayed_outvars.intersection_update(microbatch_bound_in_to_outs.keys())
    for invar in delayed_invars:
        microbatch_bound_in_to_outs[invar] = gensym_fn(invar.aval)
    # rewrite the microbatch_bound
    new_microbatch_bound_invars = []
    new_microbatch_bound_outvars = []
    for idx, var in enumerate(microbatch_bound.invars + list(delayed_invars)):
        # remove vars now defined after microbatch_bound.
        if isinstance(var, Var) and var in delayed_outvars:
            continue
        new_microbatch_bound_invars.append(var)
        # add vars now used after microbatch_bound.
        new_microbatch_bound_outvars.append(
            microbatch_bound.outvars[idx] if idx < len(microbatch_bound.invars)
            else microbatch_bound_in_to_outs[var])
    new_microbatch_bound = clone_jaxpr_eqn(microbatch_bound,
                                           new_microbatch_bound_invars,
                                           new_microbatch_bound_outvars)
    return new_microbatch_bound, microbatch_bound_in_to_outs


def _rewrite_delayed_gradient_sum_eqns(delayed_eqns,
                                       microbatch_bound_in_to_outs):
    """Change args of eqns that are delayed to the non-microbatched part."""
    new_apply_eqns = []
    for eqn in delayed_eqns:
        invars = [
            microbatch_bound_in_to_outs[var] if isinstance(var, Var) and
            var in microbatch_bound_in_to_outs else var for var in eqn.invars
        ]
        outvars = [
            microbatch_bound_in_to_outs[var] if not isinstance(var, DropVar) and
            var in microbatch_bound_in_to_outs else var for var in eqn.outvars
        ]
        new_apply_eqns.append(clone_jaxpr_eqn(eqn, invars, outvars))
    return new_apply_eqns

def _rewrite_compute_eqns(eqns, eqn_moved_to, gensym_fn):
    """Insert unmarked eqns(eqn_moved_to) to compute eqn sequence."""
    marker_cnt = 0
    new_eqns = []
    for eqn in eqns:
        if eqn.primitive is not pipeline_p:
            pass
        elif eqn.params['mark_type'] == 'start':
            cur_pipeline_start_idx = len(new_eqns)
        elif marker_cnt not in eqn_moved_to:
            marker_cnt += 1
        else:
            appended_eqns = eqn_moved_to[marker_cnt]
            i_marker = new_eqns[cur_pipeline_start_idx]
            o_marker = eqn
            layer_invar_map = {
                inv: outv
                for inv, outv in zip(i_marker.invars, i_marker.outvars)
                if isinstance(inv, Var) and not isinstance(outv, DropVar)
            }
            layer_outvar_map = {
                outv: inv
                for inv, outv in zip(o_marker.invars, o_marker.outvars)
                if isinstance(inv, Var) and not isinstance(outv, DropVar)
            }
            # collect and create all vars, then rewrite and create eqns
            inserted_invars = OrderedSet()
            inserted_outvars = OrderedSet()
            for eq in appended_eqns:
                # collect and create all used and output vars
                eq_new_invs = []
                for inv in eq.invars:
                    if isinstance(inv, Var):
                        if inv in layer_outvar_map:
                            # this layer defines the invar, use pre-marker ver.
                            eq_new_invs.append(layer_outvar_map[inv])
                        else:
                            if inv not in layer_invar_map:
                                # add new invar from other layers
                                layer_invar_map[inv] = gensym_fn(inv.aval)
                                inserted_invars.add(inv)
                            eq_new_invs.append(layer_invar_map[inv])
                    else:
                        eq_new_invs.append(inv)
                eq_new_outvs = []
                for outv in eq.outvars:
                    if isinstance(outv, DropVar):
                        eq_new_outvs.append(outv)
                    else:
                        new_mapped = gensym_fn(outv.aval)
                        layer_outvar_map[outv] = new_mapped
                        inserted_outvars.add(new_mapped)
                        eq_new_outvs.append(new_mapped)
                # create the new eqn
                new_eqns.append(clone_jaxpr_eqn(eq, eq_new_invs, eq_new_outvs))

            # create the new in marker
            new_eqns[cur_pipeline_start_idx] = _insert_to_pipeline_marker(
                i_marker, inserted_invars, layer_invar_map)
            layer_outvar_map = {v: k for k, v in layer_outvar_map.items()}
            eqn = _insert_to_pipeline_marker(o_marker, inserted_outvars,
                                             layer_outvar_map)
            marker_cnt += 1

        new_eqns.append(eqn)
    return new_eqns
def _insert_to_pipeline_marker(marker, new_inv, mapping):
    invs = list(marker.invars)
    outvs = list(marker.outvars)
    for inv in new_inv:
        invs.append(inv)
        outvs.append(mapping[inv])
    return clone_jaxpr_eqn(marker, invs, outvs)
