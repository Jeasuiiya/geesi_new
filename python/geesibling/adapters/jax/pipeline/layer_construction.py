from functools import wraps
import numpy as np
from typing import Callable, Iterable,Union


from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.api import make_jaxpr
from jax.core import (ClosedJaxpr, DropVar, Literal, gensym, jaxpr_as_fun, Jaxpr, Var)
from geesibling.adapters.jax.pipeline.primitive_def import mark_pipeline_jaxpreqn
from geesibling.adapters.jax.pipeline.util import (OrderedSet, 
                                                   clone_jaxpr, new_jaxpr_eqn, clone_jaxpr_eqn, slices_to_jaxpr,
                                                   maybe_numba_jit,
                                                   is_nontrivial,
                                                   eqn_flops,heavy_count,
                                                   get_var_mapping)

import logging

logger = logging.getLogger(__name__)
LAYER_HEAVY_OP_LOWER_BOUND = 3
DEFAULT_EPS = 0.5
DEFAULT_COST_CRITERIA = "flops"


def add_pipeline_marks_for_sliced_eqns(closed_jaxpr: ClosedJaxpr, sliced_eqns):
    """Adds pipeline marks for sliced equations."""
    layer_num = len(sliced_eqns)
    layer_pipeline_invars = [OrderedSet() for _ in range(layer_num)]
    layer_pipeline_outvars = [OrderedSet() for _ in range(layer_num)]
    var_layer_dict = {}
    var_mapping = {}

    # 对于计算图的全局输入变量，将它们的层级标记为-1，表示它们是全局输入
    # build mapping dicts for global invars
    for var in closed_jaxpr.jaxpr.invars:
        var_layer_dict[var] = -1

    # 遍历每个方程式。若输入不是常量并不在全局变量中且没被标记在这层，将其添加到当前层的输入变量集合中，若已经在之前的层出现过就添加到输出变量集合中
    # build mapping dicts for all eqns
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in closed_jaxpr.jaxpr.constvars and
                        var_layer_dict[var] != i):
                    layer_pipeline_invars[i].add(var)
                    if var_layer_dict[var] == -1:
                        continue
                    layer_pipeline_outvars[var_layer_dict[var]].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i

    # 构建全局遍历的输出映射
    # build mapping dict for global outvars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    literal_outvar_eqns = []
    literal_outvar_marker_invars = []
    literal_outvar_marker_outvars = []
    for idx, var in enumerate(closed_jaxpr.jaxpr.outvars):
        if isinstance(var, Literal):
            # add a dummy equation to transform a Literal into a normal Var
            if isinstance(var.val, np.ndarray):
                val = np.zeros_like(var.val)
            elif isinstance(var.val, Iterable):
                raise NotImplementedError()
            else:
                val = type(var.val)(0)
            zero_literal = Literal(val, var.aval)
            new_var = gensym_func(var.aval)
            new_eqn = new_jaxpr_eqn([var, zero_literal], [new_var], lax.add_p,
                                    {})
            literal_outvar_eqns.append(new_eqn)
            literal_outvar_marker_invars.append(new_var)
            literal_outvar_marker_outvars.append(gensym_func(var.aval))
            var_mapping[idx] = literal_outvar_marker_outvars[-1]
        elif var in closed_jaxpr.jaxpr.constvars or var_layer_dict[var] == -1:
            raise NotImplementedError(
                "Does not support this use case of output var.")
        else:
            layer_pipeline_outvars[var_layer_dict[var]].add(var)
    
    # 构建新的方程式，给对应的每一层的开头和结尾加上start和end和layer标记
    # build new equations
    new_eqns = []
    for i, eqns in enumerate(sliced_eqns):
        # pipeline start eqn
        computation_var_mapping = {}

        pipeline_start_invars = []
        pipeline_start_outvars = []
        for var in layer_pipeline_invars[i]:
            new_var = gensym_func(var.aval)
            pipeline_start_invars.append(get_var_mapping(var_mapping, var))
            pipeline_start_outvars.append(new_var)
            computation_var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_start_invars,
                                   pipeline_start_outvars, f"layer_{i}",
                                   "start"))
        # all other eqns
        for eqn in (eqns + literal_outvar_eqns if i == 0 else eqns):
            new_invars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.invars
            ]
            new_eqns.append(clone_jaxpr_eqn(eqn, new_invars))

        # pipeline end eqn
        pipeline_end_invars = list(
            literal_outvar_marker_invars) if i == 0 else []
        pipeline_end_outvars = list(
            literal_outvar_marker_outvars) if i == 0 else []
        for var in layer_pipeline_outvars[i]:
            new_var = gensym_func(var.aval)
            pipeline_end_invars.append(
                get_var_mapping(computation_var_mapping, var))
            pipeline_end_outvars.append(new_var)
            var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars,
                                   f"layer_{i}", "end"))

    # 重新构建输出变量
    new_outvars = []
    for idx, var in enumerate(closed_jaxpr.jaxpr.outvars):
        if isinstance(var, Literal):
            new_outvars.append(var_mapping[idx])
        else:
            new_outvars.append(get_var_mapping(var_mapping, var))
    
    # 重新生成方法的jaxpr
    new_closed_jaxpr = clone_jaxpr(closed_jaxpr,
                                   outvars=new_outvars,
                                   eqns=new_eqns)
    return new_closed_jaxpr



# slice jaxpr to sliced_eqns
def slice_eqns_by_eqnsnum(closed_jaxpr:ClosedJaxpr, layer_num:int):
    """Slices eqns by the number of eqns.
    examples:the length of eqns is 10,and the layer_num is 5,
    then each layer have two eqns.
    
    if length % layer num != 0,
    example:the length of eqns is 11,and the layer_num is 3,
    layer_size = leng // layer_num + 1, layer_size = 4
    the first and second layer have 4 eqns 
    and the last have the rest of eqns which is 3 eqns 

    sliced_eqns = []
    current_computation_eqns = []
    leng = len(closed_jaxpr.jaxpr.eqns)
    if(leng % layer_num == 0):
        layer_size = leng // layer_num
    else:
        layer_size = leng // layer_num + 1
    flag = 1
    print("OOOOOOOOOOOOO",leng,layer_size)
    for eqn in closed_jaxpr.jaxpr.eqns:
        current_computation_eqns.append(eqn)
        if (flag % layer_size == 0 ):
            sliced_eqns.append(current_computation_eqns)
            current_computation_eqns = []
        flag += 1
    if(len(current_computation_eqns) != 0):
        sliced_eqns.append(current_computation_eqns)

    """
    sliced_eqns = []
    current_computation_eqns = []
    leng = len(closed_jaxpr.jaxpr.eqns)
#    layre=[int(leng*0.1),int(leng*0.1),int(leng*0.1),leng-int(leng*0.1)-int(leng*0.1)-int(leng*0.1)]
    layre=[int(leng*0.25),int(leng*0.25),int(leng*0.25),leng-int(leng*0.25)-int(leng*0.25)-int(leng*0.25)]
    print(layre)
    flag=0
    num=0
    for eqn in closed_jaxpr.jaxpr.eqns:
        current_computation_eqns.append(eqn)
        flag+=1
        if flag==layre[num]:
            sliced_eqns.append(current_computation_eqns)
            current_computation_eqns = []
            flag=0
            num+=1
    return sliced_eqns






def jaxpr_eqns_input_sizes(jaxpr) -> np.ndarray:
    """Return a list of input sizes for each equation in the jaxpr.

    Args:
        jaxpr: Jaxpr to get input sizes for.

    Returns:
        A #eqns * #eqns numpy array of input sizes. cost[l, r] represents the
        input size of the l-th to (r - 1)-th equation in the jaxpr.
    """
    length = len(jaxpr.eqns)
    input_sizes = np.full((length + 1, length + 1), 0, dtype=np.float32)

    outvars = OrderedSet()
    for k in range(0, length + 1):
        if k > 0:
            outvars = outvars.union(jaxpr.eqns[k - 1].outvars)
        invars = OrderedSet()
        total_size = 0
        for r in range(k + 1, length + 1):
            for invar in jaxpr.eqns[r - 1].invars:
                if (isinstance(invar, Var) and invar in outvars and
                        invar not in invars):
                    invars.add(invar)
                    if isinstance(invar.aval.dtype, np.dtype):
                        total_size += invar.aval.size * invar.aval.dtype.itemsize
            input_sizes[k, r] = total_size
    return input_sizes


def get_layer_construction_costs(jaxpr, cost_criteria="flops"):
    """Gets the layer construction cost."""
    nontrivial = np.array([is_nontrivial(eqn) for eqn in jaxpr.eqns],
                          dtype=np.int32)
    input_sizes = jaxpr_eqns_input_sizes(jaxpr)
    if cost_criteria == "flops":
        compute_costs = np.array([
            eqn_flops(eqn) if nt else 0
            for nt, eqn in zip(nontrivial, jaxpr.eqns)
        ],
                                 dtype=np.float64)
    elif cost_criteria == "count":
        compute_costs = np.array([
            heavy_count(eqn) if nt else 0
            for nt, eqn in zip(nontrivial, jaxpr.eqns)
        ],
                                 dtype=np.float64)
    else:
        raise ValueError(f"Unrecoginzed cost criteria {cost_criteria}")
    return nontrivial, input_sizes, compute_costs


def cluster_jaxpr_by_cost(jaxpr: Jaxpr, layer_num: int, eps: float, costs,
                          cost_criteria):
    """Clusters the jaxpr by cost."""
    layer_num = int(layer_num)
    length = len(jaxpr.eqns)
    non_trivial, input_sizes, compute_costs = costs
    compute_costs_avg = compute_costs.sum() / layer_num
    if cost_criteria in ("flops", "input_memory"):
        compute_costs_bound = compute_costs_avg * (1 + eps)
    elif cost_criteria == "count":
        compute_costs_bound = max(compute_costs_avg * (1 + eps),
                                  compute_costs_avg + 5)
    else:
        raise ValueError(f"Unrecoginzed cost criteria {cost_criteria}")
    layer_heavy_op_lower_bound = LAYER_HEAVY_OP_LOWER_BOUND
    if sum(non_trivial) / layer_num < layer_heavy_op_lower_bound:
        layer_heavy_op_lower_bound = int(sum(non_trivial) / layer_num)  # noqa
        logger.warning(
            "Too few non-trivial ops (dot, conv), which may influence"
            " auto-sharding performance")

    @maybe_numba_jit
    def init():
        blocked = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
        for left in range(1, length + 1):
            cnt = 0
            total_compute_cost = 0
            for r in range(left, length + 1):
                if non_trivial[r - 1]:
                    cnt += 1
                    total_compute_cost += compute_costs[r - 1]
                if cnt < layer_heavy_op_lower_bound:
                    if total_compute_cost >= compute_costs_bound:
                        blocked[left, r] = 0
                    continue
                if (total_compute_cost >= compute_costs_bound and
                        non_trivial[r - 1] and
                        cnt > layer_heavy_op_lower_bound):
                    break
                blocked[left, r] = 0
        return blocked

    @maybe_numba_jit
    def dp(input_sizes, blocked):
        max_cost = np.full((length + 1, layer_num + 1),
                           np.inf,
                           dtype=np.float32)
        sum_cost_under_max = np.full((length + 1, layer_num + 1),
                                     np.inf,
                                     dtype=np.float32)
        max_cost_argmin = np.full((length + 1, layer_num + 1),
                                  -1,
                                  dtype=np.int32)
        solution_imbalance = np.full((length + 1, layer_num + 1),
                                     np.inf,
                                     dtype=np.float32)
        max_cost[0, 0] = 0
        sum_cost_under_max[0, 0] = 0
        # Currently use variance to measure imbalance
        for r in range(0, length + 1):
            solution_imbalance[r, 0] = 0

        for q in range(1, layer_num + 1):
            for r in range(1, length + 1):
                for k in range(0, r):
                    new_value = max(max_cost[k, q - 1],
                                    blocked[k + 1, r] + input_sizes[k, r])
                    new_sum = (sum_cost_under_max[k, q - 1] +
                               blocked[k + 1, r] + input_sizes[k, r])
                    new_imbalance = (solution_imbalance[k, q - 1] + k**2 / q -
                                     r**2 / (q + 1) + (r - k)**2)
                    if (new_value < max_cost[r, q] or
                        (new_value <= max_cost[r, q] * (1 + 1e-4) and
                         (new_sum < sum_cost_under_max[r, q] or
                          (new_sum <= sum_cost_under_max[r, q] * (1 + 1e-4) and
                           new_imbalance < solution_imbalance[r, q])))):
                        max_cost[r, q] = new_value
                        sum_cost_under_max[r, q] = new_sum
                        max_cost_argmin[r, q] = k
                        solution_imbalance[r, q] = new_imbalance
        return max_cost_argmin, max_cost[length, layer_num]

    blocked = init()
    a_argmin, value = dp(input_sizes, blocked)

    reversed_sliced_eqns = []

    r = length
    for q in range(layer_num, 0, -1):
        k = a_argmin[r, q]
        reversed_sliced_eqns.append(jaxpr.eqns[k:r])
        r = k
    assert r == 0, "No solution for layer construction."
    solution = list(reversed(reversed_sliced_eqns))

    # print("dp solution")
    # for i, eqns in enumerate(solution):
    #    invars = OrderedSet()
    #    for eqn in eqns:
    #        invars.update([var for var in eqn.invars if isinstance(var, Var)])
    #    invars.intersection_update(jaxpr.jaxpr.invars)
    #    print(f"mesh: {i},  set_shapes: "
    #          f"{[x.aval.shape for x in invars if len(x.aval.shape) > 1]}")
    #
    #    invars = []
    #    for eqn in eqns:
    #        tmp_set = set([var for var in eqn.invars if isinstance(var, Var)])
    #        tmp_set.intersection_update(jaxpr.jaxpr.invars)
    #        invars.extend(list(tmp_set))
    #    print(f"mesh: {i}, list_shapes: "
    #          f"{[x.aval.shape for x in invars if len(x.aval.shape) > 1]}")

    solution_info = {
        "total_cost": value,
    }
    return solution, solution_info


def search_layer_num(jaxpr,
                     eps,
                     layer_eps=0,
                     cost_criteria=DEFAULT_COST_CRITERIA):
    """TODO(zhuohan): docstring."""
    non_trivial, input_sizes, compute_costs = get_layer_construction_costs(
        jaxpr)
    layer_num = 2
    r = int(non_trivial.sum() / 3) + 1
    _, solution_info = cluster_jaxpr_by_cost(
        jaxpr,
        layer_num,
        eps, (non_trivial, input_sizes, compute_costs),
        cost_criteria=cost_criteria)
    l_val = solution_info["total_cost"]
    while r - layer_num > 1:
        mid = int((layer_num + r) / 2)
        _, solution_info = cluster_jaxpr_by_cost(
            jaxpr,
            mid,
            eps, (non_trivial, input_sizes, compute_costs),
            cost_criteria=cost_criteria)
        mid_val = solution_info["total_cost"]
        if mid_val > l_val * (1 + layer_eps):
            r = mid
        else:
            layer_num = mid
    return layer_num



def layer_level_transformation(fun: Callable, 
                               layer_num: Union[int, str] = None,
                               eps: float = DEFAULT_EPS,
                               cost_criteria: str = DEFAULT_COST_CRITERIA,
                               layer_eps: float = 0.0):
    
    def decorate_fun(fun):
        @wraps(fun)
        def wrapped(*args):
            jaxpr, out_shape_tree = make_jaxpr(fun,
                                            static_argnums=(),
                                            return_shape=True)(*args)
            
            nonlocal layer_num
            if layer_num == "auto" or layer_num == None:
                #nonlocal layer_num
                layer_num = search_layer_num(jaxpr, eps, layer_eps)
            costs = get_layer_construction_costs(jaxpr,
                                                 cost_criteria=cost_criteria)
    #        sliced_eqns, _ = cluster_jaxpr_by_cost(jaxpr,
     #                                              layer_num,
      #                                             eps,
       #                                            costs,
        #                                           cost_criteria=cost_criteria)
            #print("sliced_eqns length:",len(sliced_eqns[0]),len(sliced_eqns[1]),len(sliced_eqns[2]),len(sliced_eqns[3]))
            
            sliced_eqns = slice_eqns_by_eqnsnum(jaxpr,layer_num)
            
            jaxpr = add_pipeline_marks_for_sliced_eqns(jaxpr, sliced_eqns)

            flatten_args, _ = tree_flatten(args)
            ans = jaxpr_as_fun(jaxpr)(*flatten_args)  # pylint: disable=not-callable
            _, out_tree = tree_flatten(out_shape_tree)
            return tree_unflatten(out_tree, ans)

        return wrapped

    if fun is None:
        return decorate_fun
    else:
        return decorate_fun(fun)



def split_pr(pr):
    layer_num = 4
    
    #sliced_eqns = slice_eqns_by_eqnsnum(pr, layer_num)
    
    #sliced_jaxprs,in_connects = slices_to_jaxpr(pr, sliced_eqns)

    return layer_num
