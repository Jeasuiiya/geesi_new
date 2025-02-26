from typing import Any, Dict, List
from geesibling.core.types import Block, GraphPortRef, BlockPortRef, BlockId, SubGraph
from geesibling.tools import log

__doc__ = "schedule blocks"

#表示调度上下文的状态，跟踪和管理调度上下文的状态，其中包括图，块，输入变量，返回值等映射关系和列表
class ScheduleContext:
    """
    schedule context status
    """

    graph2block = Dict[SubGraph, Block]
    block2graph = Dict[Block, SubGraph]
    entry_blocks = List[Block]
    invars = List[GraphPortRef]
    returns = List[GraphPortRef]
    block_input_var = List[BlockPortRef]
    block_outputs = Dict[BlockId, List]
    nodeoutput_blockoutput = Dict[GraphPortRef, BlockPortRef]
    blockoutput_nodeoutput = Dict[BlockPortRef, GraphPortRef]
    execute_successor = Dict[Block, List[Block]]
    node_output_type = Dict[GraphPortRef, Any]

    def __init__(self, invars, returns, node_output_type, out_tree, cache_block_executable):
        self.invars = invars
        self.returns = returns
        self.node_output_type = node_output_type
        self.graph2block = {}
        self.block2graph = {}
        self.entry_blocks = []
        self.block_outputs = {}
        self.block_input_var = [None] * len(invars)
        self.nodeoutput_blockoutput = {}
        self.blockoutput_nodeoutput = {}
        self.execute_successor = {}
        self.out_tree = out_tree
        self.cache_block_executable = cache_block_executable

    def order(self):
        """
        return blocks with topo order
        """
        queue = []
        ready = set({(0, i) for i in range(len(self.invars))})
        visited = set()
        #判断块是否可以进入队列
        def can_enqueue(b):
            if b.inputports_size == 0 and b.outputports_size >= 0:
                return True and b not in visited
            if b.inputports_size == 0 and b.outputports_size == 0:
                return False
            complete_flag = 0
            in_ports = b.inputports
            for i in map(lambda x: (x[1], x[2]), in_ports):
                if i in ready:
                    complete_flag = complete_flag + 1
            if complete_flag == len(in_ports):
                return True and b not in visited
            return False

        def enqueue(block, callback=None):
            if can_enqueue(block):
                callback(b)
                visited.add(block)
        #初始化队列的第一层
        queue_level = []
        for b in self.block2graph:
            enqueue(b, queue_level.append)
        queue.append(queue_level)
        ready = set({(0, i) for i in range(len(self.invars))})

        while len(queue) != 0:
            queue_level = queue[0]
            queue.pop(0)
            yield queue_level
            for current_block in queue_level:
                # yield current_block
                for i in current_block.outputports:
                    ready.add((current_block.id, i[0]))
            next_level = []
            for current_block in queue_level:
                for b in self.execute_successor.get(current_block, ()):
                    enqueue(b, next_level.append)
            if len(next_level) > 0:
                queue.append(next_level)
        # assert len(self.block2graph) == count

        not_visited = []
        for b, g in self.block2graph.items():
            if b not in visited:
                not_visited.append((b.id, tuple(map(lambda x: self.graph2block[x].id, g.input_graphs))))
        if not_visited:
            log.debug("not visited block: %s", not_visited)

    def block(self, graph: SubGraph):
        """Manage a graph"""
        if not isinstance(graph, SubGraph):
            graph = SubGraph(graph)
        b = Block(graph)
        # record ref 记录引用的关系
        assert self.graph2block.get(graph, None) is None
        log.trace("block: %s nodes: %s", b.id, (i.name for i in graph.nodes))
        self.graph2block[graph] = b
        self.block2graph[b] = graph
        return b

    def blocks(self, graphs: List[SubGraph]):
        """
        register subgraphs as blocks
        """
        for i in graphs:
            self.block(i)
        
    #准备图的输出，为每个块添加输出端口，并建立节点输出引用与块端口引用这间的映射关系，为了在调度执行时能够正确地将输出传给下一个块  单块的输出以及整体的输出
    def _prepare_outputs(self):
        return_node_names = []
        if self.returns is not None and len(self.returns) != 0:
            return_node_names = list(zip(*self.returns))[0]
        log.debug("returns: %s", self.returns)
        for block in self.graph2block.values():
            graph = block.graph
            out = list(map(lambda x: self.graph2block[x], graph.output_graphs))
            log.trace("block: %s output blocks: %s", block.id, out)
            self.execute_successor[block] = out#将当前块的后继设置为图的输出块
            for j in range(graph.nodes_num):
                node = graph.get_node(j)
                if node.name in return_node_names:
                    for k in node.output_indexes():
                        if node.output_name(k) in self.returns:
                            block.add_outputport(node.output_type(k), node.output_shape(k))
                            # -1 -> last line add
                            block_ref = BlockPortRef(block.id, block.outputports[-1].index)
                            node_ref = node.output_name(k)
                            self.nodeoutput_blockoutput[node_ref] = block_ref
                            self.blockoutput_nodeoutput[block_ref] = node_ref

            for output_maps in graph.outputs:
                for k in output_maps:
                    node = graph.get_node(k.this.node)
                    node_ref = node.output_name(k.this.index)
                    if self.nodeoutput_blockoutput.get(node_ref, None) is None:
                        block.add_outputport(node.output_type(k.this.index), node.output_shape(k.this.index))
                        block_ref = BlockPortRef(block.id, block.outputports[-1].index)
                        self.nodeoutput_blockoutput[node_ref] = block_ref
                        self.blockoutput_nodeoutput[block_ref] = node_ref

    def _prepare_inputs(self):
        input_node_names = []
        if self.invars is not None and len(self.invars) != 0:
            input_node_names = list(zip(*self.invars))[0]
        for block in self.graph2block.values():
            graph = block.graph
            for node_index in range(graph.nodes_num):
                node = graph.get_node(node_index)
                if node.op == "Input" and node.name in input_node_names:
                    if block not in self.entry_blocks:
                        self.entry_blocks.append(block)
                    global_input_index = input_node_names.index(node.name)
                    block.add_inputport(0, global_input_index, node.output_type(0), node.output_shape(0))

                    self.block_input_var[global_input_index] = BlockPortRef(block.id, len(block.inputports) - 1)
                    self.blockoutput_nodeoutput[(0, global_input_index)] = node.output_name(0)
            input_graphs, inputs = graph.input_graphs, graph.inputs
            assert len(input_graphs) == len(inputs)

            for g, input_maps in zip(graph.input_graphs, graph.inputs):
                ref_block = self.graph2block[g]
                ref_block_id = ref_block.id
                node_ref_set = set()
                for k in input_maps:
                    if k.pre in node_ref_set:
                        continue
                    node_ref_set.add(k.pre)
                    node = g.get_node(k.pre.node)
                    block.add_inputport(
                        ref_block_id,
                        self.nodeoutput_blockoutput[node.output_name(k.pre.index)].index,
                        node.output_type(k.pre.index),
                        node.output_shape(k.pre.index),
                    )

    def regular_blocks(self):
        self._prepare_outputs()
        self._prepare_inputs()
