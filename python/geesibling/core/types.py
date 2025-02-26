"""
wrapper block
"""
from collections import namedtuple
from geesibling.core.lib._graph import Block as RawBlock, Graph as RawGraph, Node as RawNode, SubGraph as RawSubGraph

NodeName = str
BlockId = int
PortIndex = int

InputPortRef = namedtuple("InputPortRef", ["index", "source", "source_index", "dtype", "shape"])
OutputPortRef = namedtuple("OutputPortRef", ["index", "dtype", "shape"])

GraphPortRef = namedtuple("GraphPortRef", ["node", "index"])  # Tuple[NodeName, int]
BlockPortRef = namedtuple("BlockPortRef", ["block", "index"])  # Tuple[Block, int]

# Tuple[PortIndex, BlockId, int, DataType, List]
BlockInputPort = namedtuple("BlockInputPort", ["index", "source", "source_index", "dtype", "shape"])

BlockOutPort = namedtuple("BlockOutPort", ["index", "dtype", "shape"])  # Tuple[PortIndex, DataType, List]

SubGraphInputMap = namedtuple("SubGraphInputMap", ["pre", "this"])
SubGraphOutputMap = namedtuple("SubGraphOutputMap", ["this", "next"])


class Block(RawBlock):
    """Wrapper block"""

    @property
    def graph(self):
        return SubGraph(super().graph)

    @property
    def inputports(self):
        return [InputPortRef(*i) for i in super().inputports]

    @property
    def outputports(self):
        return [OutputPortRef(*i) for i in super().outputports]


class Graph(RawGraph):
    """Wrapper graph"""

    @property
    def returns(self):
        return [GraphPortRef(*i) for i in super().returns]

    def get_node(self, *args, **kwargs):
        return Node(super().get_node(*args, **kwargs))


class Node(RawNode):
    """Wrapper node"""

    def input_name(self, *args, **kwargs):
        return GraphPortRef(super().input_name(*args, **kwargs))

    def output_name(self, *args, **kwargs):
        return GraphPortRef(super().output_name(*args, **kwargs))


class SubGraph(RawSubGraph):
    """Wrapper subgraph"""

    @property
    def returns(self):
        return [GraphPortRef(*i) for i in super().returns]

    @property
    def inputs(self):
        return [
            [SubGraphInputMap(*[GraphPortRef(*m) for m in map]) for map in graph_map] for graph_map in super().inputs
        ]

    @property
    def outputs(self):
        return [
            [SubGraphOutputMap(*[GraphPortRef(*m) for m in map]) for map in graph_map] for graph_map in super().outputs
        ]

    @property
    def input_graphs(self):
        return [SubGraph(g) for g in super().input_graphs]

    @property
    def output_graphs(self):
        return [SubGraph(g) for g in super().output_graphs]
