from copy import deepcopy
import itertools as it
from typing import Dict
from dataclasses import dataclass
import numpy as np
import jax
import jax.core as jc
import jax.lax as jl
import jax._src.prng as jprng
import jax._src.ad_util as jad_util
import jax._src.pjit as jpjit
#from jax.experimental import pjit as jpjit
#from jax.interpreters.xla import xla_call
import jax._src.state as jstate
from jax import util
from geesibling.core.lib._graph import DataType
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
float8_e4m3fn = ml_dtypes.float8_e4m3fn
float8_e5m2 = ml_dtypes.float8_e5m2

__doc__ = """
convert graph to jaxpr
Author: yiguangzheng
datetime: 2023.7.4
version: 1 2023.7.4 first commit
"""

DATA_TYPE_TO_SHAPE_ARRAY_DTYPE = {
    DataType.BOOL: np.dtype("bool"),
    DataType.I8: np.dtype("int8"),
    DataType.I16: np.dtype("int16"),
    DataType.I32: np.dtype("int32"),
    DataType.I64: np.dtype("int64"),
    DataType.U8: np.dtype("uint8"),
    DataType.U16: np.dtype("uint16"),
    DataType.U32: np.dtype("uint32"),
    DataType.U64: np.dtype("uint64"),
    DataType.F8E4M3FN: np.dtype(float8_e4m3fn),
    DataType.F8E5M2: np.dtype(float8_e5m2),
    DataType.BF16: np.dtype(bfloat16),
    DataType.F16: np.dtype("float16"),
    DataType.F32: np.dtype("float32"),
    DataType.F64: np.dtype("float64"),
}


OPS = {}


def register_op(*ops):
    for op in ops:
        if isinstance(op, jc.Primitive):
            OPS[op.name] = op


def register_module_ops(m):
    for i in dir(m):
        p = getattr(m, i)
        register_op(p)


register_module_ops(jl)
#register_module_ops(xla_call)
register_module_ops(jax.custom_derivatives)
register_module_ops(jprng)
register_module_ops(jad_util)
register_module_ops(jpjit)
register_module_ops(jstate)


@dataclass
class ConvertContext:
    source_info: jc.source_info_util.SourceInfo
    counter: it.count
    output_var: Dict[str, jc.Var]
    #input_var: Dict[str, jc.Var]

def topo_graph(graph, indegree0, visited):
    """
    return nodes as topo order
    """
    visited = deepcopy(visited)
    queue = [*indegree0]

    def can_enquque(node_name):
        if node_name in visited or node_name in queue:
            return False
        next_node_inputs = graph.get_node(node_name).inputs
        complete_flag = 0
        for i in next_node_inputs:
            if i in visited:
                complete_flag = complete_flag + 1
            else:
                return False
        if complete_flag == len(next_node_inputs):
            return True
        return False

    def enqueue(node_name):
        if can_enquque(node_name):
            queue.append(node_name)

    for input_maps in graph.inputs:
        for m in input_maps:
            enqueue(m[1][0])
    while len(queue) != 0:
        current_node_name = queue[0]
        queue.pop(0)
        current_node = graph.get_node(current_node_name)
        next_nodes = []
        if current_node_name not in visited:
            if current_node.op not in ["Input", "Const", "ConstVar"]:
                yield current_node_name
            visited.add(current_node_name)
        next_nodes = current_node.outputs
        for next_node_name in next_nodes:
            next_node = graph.get_node(next_node_name)
            if next_node is None:
                continue
            enqueue(next_node.name)


SUFFIX = ""


def _prepare_const(context, graph, write):
    vars_const = {}
    indegree0 = []
    for i in range(graph.nodes_num):
        current_node = graph.get_node(i)
        op = current_node.op
        if len(current_node.input_indexes()) == 0:
            indegree0.append(current_node.name)
        if op in ["Const", "ConstVar"]:
            for j in current_node.output_indexes():
                data_type, data_shape = current_node.output_type(j), current_node.output_shape(j)
                if op == "Const":
                    val = current_node.output_value(j)
                    assert val is not None  # output of Const must has value
                    if val.dtype != DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type]:
                        val = val.view(DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type])
                    if val.shape != data_shape:
                        val = val.reshape(data_shape)
                    write(current_node.output_name(j), jc.Literal(val, jax.xla.abstractify(val)))
                elif op == "ConstVar":
                    vars_const[current_node.output_name(j)] = jc.Var(
                        next(context.counter),
                        SUFFIX,
                        jc.ShapedArray(data_shape, DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type]),
                    )
    return vars_const, indegree0


def _prepare_invars(context, scontext, block):
    vars_in = {}
    for i in block.inputports:
        node_ref = scontext.blockoutput_nodeoutput[(i.source, i.source_index)]
        data_type, data_shape = i.dtype, i.shape
        if data_type == DataType.Other:
            data_type = scontext.node_output_type[node_ref]
        else:
            data_type = DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type]
        vars_in[node_ref] = jc.Var(
            next(context.counter), SUFFIX, jc.ShapedArray(data_shape, data_type)
            )
    return vars_in


def block2jaxpr(sctx, block, params=None, inline=False):
    """
    convert block to jaxpr
    """
    graph = block.graph
    if params is None:
        params = {}
    ctx = ConvertContext(
        source_info=jc.source_info_util.SourceInfo(None, jc.source_info_util.NameStack()),
        counter=it.count(),
        output_var={},
    )

    def read(name):
        return ctx.output_var[name]

    def write(name, var):
        ctx.output_var[name] = var

    vars_const, indegree0 = _prepare_const(ctx, graph, write)

    vars_in = _prepare_invars(ctx, sctx, block)

    util.safe_map(write, vars_in.keys(), vars_in.values())
    util.safe_map(write, vars_const.keys(), vars_const.values())
    eqns = []
    visited = list(zip(*vars_in.keys()))
    if len(visited) > 0:
        visited = set(visited[0])
    else:
        visited = set()

    def input_var(*args):
        node, i = args
        data_type, data_shape = node.input_type(i), node.input_shape(i)
        if data_type == DataType.Other:
            data_type = sctx.node_input_type[node.input_name(i)]
        else:
            data_type = DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type]
        return jc.Var(next(ctx.counter), "", jc.ShapedArray(data_shape, data_type))

    def output_var(*args):
        node, i = args
        data_type, data_shape = node.output_type(i), node.output_shape(i)
        if data_type == DataType.Other:
            data_type = sctx.node_output_type[node.output_name(i)]
        else:
            data_type = DATA_TYPE_TO_SHAPE_ARRAY_DTYPE[data_type]
        return jc.Var(next(ctx.counter), "", jc.ShapedArray(data_shape, data_type))

    for i in topo_graph(graph, indegree0, visited):
        current_node = graph.get_node(i)
        input_refs = util.safe_map(current_node.input_ref, current_node.input_indexes())
        invars = util.safe_map(read, input_refs)

        outvars = util.safe_map(
            output_var, [current_node] * len(current_node.output_indexes()), current_node.output_indexes()
        )

        p = params.get(current_node.name, {})
        if p.get("inline", None) is False:
            p["inline"] = inline
        eqn = jc.JaxprEqn(
            invars=invars,
            outvars=outvars,
            primitive=OPS[current_node.op],
            params=p,
            effects=None,
            source_info=ctx.source_info,
        )
        eqns.append(eqn)
        util.safe_map(write, util.safe_map(current_node.output_name, current_node.output_indexes()), outvars)
    outvars = []
    for i in block.outputports:
        outvars.append(read(sctx.blockoutput_nodeoutput[(block.id, i[0])]))
    const_vars = list(zip(*vars_const.items()))
    pr = jc.Jaxpr(
        constvars=[] if len(const_vars) == 0 else const_vars[1],
        invars=util.safe_map(read, vars_in.keys()),
        outvars=outvars,
        eqns=eqns,
    )

    return pr, [] if len(const_vars) == 0 else const_vars[0]
