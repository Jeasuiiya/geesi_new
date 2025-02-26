"""
File: graph_dot.py
Author: yiguangzheng
Description: export geesibling graph
"""
from geesibling.core.lib._graph import Graph
import pydot


def from_graph(graph: Graph, blocks=None):
    """export from graph"""
    G = pydot.Graph()
    id_Node = {n.name: pydot.Node(n.name) for n in graph.nodes}
    unknown = pydot.Node("unknown")
    for v in id_Node.values():
        G.add_node(v)
    for i in graph.nodes:
        for output in i.outputs:
            G.add_edge(pydot.Edge(id_Node[i.name], id_Node.get(output, None) or unknown))

    if blocks:
        for b in blocks:
            sub = pydot.Subgraph(f"cluster_{b.id}")
            sub.set_label(f"{sub.get_name()} {b.device}")  # pylint: disable=no-member
            sub.set_bgcolor("lightgrey")  # pylint: disable=no-member
            for n in b.graph.nodes:
                sub.add_node(id_Node[n.name])
            G.add_subgraph(sub)
    return G


def write_dot_graph(graph: pydot.Graph, path):
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(graph.to_string())


def from_blocks(blocks):
    """export from blocks"""
    G = pydot.Graph()
    id_subgraph = {b.id: pydot.Node(b.id) for b in blocks}
    id_subgraph[0] = pydot.Node(0)
    for b in blocks:
        G.add_node(id_subgraph[b.id])
    linked = set()
    for b in blocks:
        for inputport in b.inputports:
            link = (inputport.source, b.id)
            if link not in linked:
                linked.add(link)
                G.add_edge(pydot.Edge(id_subgraph[inputport.source], id_subgraph[b.id]))

    return G
