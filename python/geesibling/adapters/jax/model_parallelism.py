#import logging

import functools
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
from jax._src import prng
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from geesibling.adapters.jax.jaxpr2graph import jaxpr2graph
from geesibling.adapters.jax.block2jaxpr import block2jaxpr
from geesibling.core.lib._graph import divide_graph, search_policy
import jax.numpy as jnp
from geesibling.adapters.jax.schedule import ScheduleContext
from geesibling.tools import log
import time


DEVICE_MAP = {"": jax.devices("cpu")[0] if len(jax.devices("gpu")) == 0 else jax.devices("gpu")[0]}


def register_device():
    for i in jax.devices("cpu"):
        DEVICE_MAP[str(i)] = i
    for i in jax.devices("gpu"):
        DEVICE_MAP[str(i)] = i
    log.debug(DEVICE_MAP)


register_device()


#表示调度上下文，用于保存并行函数调度的参数和配置信息
#给定函数、设备和调度策略，创建一个调度上下文，其中包含了jax语法树、图像表示、设备分配哦情况等，来支持函数的并行化调度
class MakeScheduleContext:
    """
    schedule context.
    used for saving arguments for parallelizing function
    """

    def __init__(self, func, devices=(), policy="fddps", method="") -> None:
        self.func = func
        self.devices = devices
        self.device_lists = []
        self.flag = 0
        self.policy = policy
        self.args = None
        self.kwargs = None
        self.method = method

    def a(self, args, kwargs):
        self.args, self.kwargs = args, kwargs
    
    # spilt devices by layer_num
    def spilt_devices(self, layer_num):
        size = len(self.devices) / layer_num
        temp = []
        for device in self.devices:
            temp.append(device)
            if(len(temp) == size):
                self.device_lists.append(temp)
                temp = []
        
   # @functools.lru_cache()#实现对实例调用结果的缓存
    def __call__(self, *input):#使得实例对象可以像函数一样被调
        pr,devices=input[0][0],input[0][1]
        self.devices=devices
        gw = jaxpr2graph(pr)
        log.debug("jaxpr2graph finished")
        g = gw.graph#使用函数将jax语法树用图像表述
        # call strategy search
        search_start_time = time.time()
        device_map = search_policy(g, self.devices, self.policy)
        search_time = time.time() - search_start_time
        #print(f"search_policy used {search_time}")
        log.debug("search policy finished. placement: %s", device_map)
        #print("device_map:",device_map)
        #搜索到的设备，将图中节点的设备信息进行更新
        if device_map is not None:
            for k, v in device_map.items():
                g.get_node(k).device = v
        else:
            log.warning("search policy failed.")
        sub_graphs = divide_graph(g)
        # prepare context
        @functools.lru_cache()
        def cache_executable(ctx, block):#用于缓存块的可执行函数
            pr, const_names = block2jaxpr(ctx, block, gw.params)#将块转化为jax语法树，并使用jax.jit对其进行编译
            const = list(map(lambda x: gw.node_ref_const[x], const_names))
            return jax.jit(functools.partial(jax.core.eval_jaxpr, pr, const), device=DEVICE_MAP[block.device])
        
        ctx= ScheduleContext(gw.invars, gw.returns, gw.node_output_type, "", cache_executable)
        ctx.blocks(sub_graphs)
        ctx.regular_blocks()#调用调度上下文的方法，对子图进行块的组织和处理
        # ctx.topo_order = tuple(filter(lambda b: b.outputports_size != 0, ctx.order())) 
        ctx.topo_order = tuple(ctx.order())#调用上下文的拓扑排序排序
        log.debug(
            "scheduled: %s, all blocks: %s ",
            functools.reduce(lambda a, b: a + len(b), ctx.topo_order, 0),
            len(ctx.graph2block),
        )

        return ctx #返回创建的调度上下文


    def get_model_parallelism_result(self, ctx, flat_args, out_tree):

        def exec_block(ctx, flat_args, block):#该函数用于执行单个块操作，通过检查块的输入端口，将输入参数进行构造，然后执行块
            bargs=[]
            for i in block.inputports:
                if i.source == 0:  # block id is 0, global input
                    bargs.append(flat_args[i.source_index])
                else:
                    a = ctx.block_outputs[i.source][i.source_index]
                    if not isinstance(a, np.ndarray) and not isinstance(a, prng.PRNGKeyArray):
                        if a.device() is not DEVICE_MAP[block.device]:
                            a = jax.device_put(a, DEVICE_MAP[block.device]).block_until_ready()
                    bargs.append(a)
            with jax.default_device(DEVICE_MAP[block.device]):
                return jax.block_until_ready(ctx.cache_block_executable(ctx, block)(*bargs))
        def schedule_level(ctx, level, flat_args):#用于调度一层中的所有块的执行，通过使用  实现并行执行，将每个块的执行任务提交给线程池
            # todo(huangchengchuang): all block in level could be parallelized
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_results = {executor.submit(exec_block, ctx, flat_args, block): block for block in level}
                for future in as_completed(future_to_results):
                    block = future_to_results[future]
                    ctx.block_outputs[block.id] = future.result()
        def returns(r):
                block_ref = ctx.nodeoutput_blockoutput[r]
                return ctx.block_outputs[block_ref.block][block_ref.index]

        for level in ctx.topo_order:
                schedule_level(ctx, level, flat_args)
        if self.method=="PipeshardParallel":
            result=[]
            for i in ctx.returns:
                result.append(returns(i))
            return result
        else:

            ctx.out_tree = out_tree
            _, out_tree = tree_flatten(ctx.out_tree)
            return tree_unflatten(out_tree, map(returns, ctx.returns))


