import functools
import time
import json
import os
import contextlib
import jax
from jax import core as jc
import numpy as np
import ml_dtypes

__doc__ = "profile op data"

CACHE_PATH = os.environ.get("GEESIBLING_PROFILE_CACHE", None) or "cache.json"
PROFILE_CACHE = None
UPDATED = False

bfloat16 = ml_dtypes.bfloat16
float8_e4m3fn = ml_dtypes.float8_e4m3fn
float8_e5m2 = ml_dtypes.float8_e5m2
SHAPE_ARRAY_DTYPE_TO_BYTES = {
    np.dtype("bool"): 1,
    np.dtype("int8"): 1,
    np.dtype("int16"): 2,
    np.dtype("int32"): 4,
    np.dtype("int64"): 8,
    np.dtype("uint8"): 1,
    np.dtype("uint16"): 2,
    np.dtype("uint32"): 4,
    np.dtype("uint64"): 8,
    np.dtype(float8_e4m3fn): 1,
    np.dtype(float8_e5m2): 1,
    np.dtype(bfloat16): 2,
    np.dtype("float16"): 2,
    np.dtype("float32"): 4,
    np.dtype("float64"): 8,
}

#初始化性能分析数据缓存的函数
def init():
    global PROFILE_CACHE  # pylint: disable=global-statement
    if PROFILE_CACHE is not None:
        return
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, mode="r", encoding="utf-8") as f:
            PROFILE_CACHE = json.load(f)
    else:
        PROFILE_CACHE = {}

#将性能分析数据缓存写回文件的函数
def write_back():
    global PROFILE_CACHE, UPDATED  # pylint: disable=global-statement
    if UPDATED:
        with open(CACHE_PATH, mode="w", encoding="utf-8") as f:
            json.dump(PROFILE_CACHE, f)
    PROFILE_CACHE = None
    UPDATED = False

#用于更新性能分析数据缓存中的特定操作
def update(op_name, input_sign, cost):
    global UPDATED  # pylint: disable=global-statement
    if PROFILE_CACHE.get(op_name, None) is None:
        op_cache = {}
        PROFILE_CACHE[op_name] = op_cache
    else:
        op_cache = PROFILE_CACHE[op_name]
    op_cache[input_sign] = cost
    UPDATED = True

#确保了性能分析期间的适当数据管理
@contextlib.contextmanager
def profile():
    init()
    yield
    write_back()

#对单个方程进行性能分析的函数
def profile_eqn(eqn):
    """
    profile single eqn
    """
    must_profile_op = ["custom_jvp_call", "pjit"]

    # make jaxpr
    def convert(var):
        if isinstance(var, jc.Literal):
            return var.aval
        return var

    def shape_dtype(var):
        return var.aval.shape, var.aval.dtype
        # return var

    op_p = eqn.primitive
    if op_p.name.startswith("random"):
        return 10, 0, 0
    pr_in = tuple(map(convert, eqn.invars))
    input_sig = list(map(shape_dtype, eqn.invars))
    input_sig_str = str(input_sig)
    if (
        PROFILE_CACHE.get(op_p.name, None) is not None
        and op_p.name not in must_profile_op
        and PROFILE_CACHE[op_p.name].get(input_sig_str, None) is not None
    ):
        return PROFILE_CACHE[op_p.name][input_sig_str]

    jaxpr = jc.Jaxpr(constvars=[], invars=pr_in, outvars=eqn.outvars, eqns=[eqn])

    # make data
    def data(var):
        if isinstance(var, jc.Literal):
            return var.val
        if isinstance(var.aval, jc.ShapedArray):
            rand = np.random.rand(*var.aval.shape)
            return np.array(rand, var.aval.dtype)
        return None

    invars = list(map(data, eqn.invars))
    start = time.time()
    jitted = jax.jit(functools.partial(jc.eval_jaxpr, jaxpr, []))#通过jit编译方程式并多次运行获取平均耗时
    for _ in range(100):
        jitted(*invars)
    end = time.time()
    cost = (end - start) * 10 * 1000
    output_sig = list(map(shape_dtype, eqn.outvars))
    output_memory = sum(
        map(
            lambda x: functools.reduce(lambda a, b: a * b, x[0], 1) * SHAPE_ARRAY_DTYPE_TO_BYTES.get(x[1], 8),
            output_sig,
        )
    )
    input_memory = sum(
        map(
            lambda x: functools.reduce(lambda a, b: a * b, x[0], 1) * SHAPE_ARRAY_DTYPE_TO_BYTES.get(x[1], 8), input_sig
        )
    )
    ret = (int(cost), input_memory, output_memory)
    if op_p.name in must_profile_op:
        return ret
    update(op_p.name, input_sig_str, ret)
    # op_cache[input_sig_str] = ret
    return ret
