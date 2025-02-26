#ifndef GEESIBLING_IR_DivideGraph_H
#define GEESIBLING_IR_DivideGraph_H

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

#include "common/error.hpp"
#include "common/result_macro.hpp"
#include "graph.hpp"
#include "node.hpp"

namespace geesibling {

using SimpleGraph = std::map<int, std::set<int>>;

/**
 * @class SplitNodeInfo
 * @brief Maintain extra info for subgraph.
 *
 */
class SplitNodeInfo {
  public:
    NodePtr node;
    bool graph_in = false;                     // 默认不是graph in算子
    bool graph_out = false;                    // 默认不是graph out算子
    std::set<std::string> same_graph_inputs;   // 同设备的前驱算子
    std::set<std::string> same_graph_outputs;  // 同设备的后继算子
    int subgraph_num = -1;                     // 默认为-1,若为-1则表示该算子没有被搜索过；
};

/**
 * @class DivideGraphHelper
 * @brief Helper for converting the whole graph to subgraphs.
 *
 */
class DivideGraphHelper {
  private:
    // name -> node info
    std::map<std::string, SplitNodeInfo> node_infos;
    // simple graph id -> node set
    std::map<int, std::set<NodePtr>> graphs;

    bool merge_enable = true;

    uint64_t merge_threshold = 100;

    /**
     * @brief For every node, simply assign graph by area.
     */
    void AssignGraph();

    /**
     * @brief Merge all node from source set to target.
     *
     * @param source Source graph nodes.
     * @param target_num Target graph simple id.
     * @param target Target graph set.
     */
    void MergeInternal(std::set<NodePtr>& source, int target_num, std::set<NodePtr>* target);

    /**
     * @brief Split graph starting from a graph out node.
     * In reversed topo order, all segment between graph out will be splited as a new graph.
     *
     * @param node Graph out node(as order start node)
     * @param nodes Graph nodes
     * @param graph_out_set A set record all grapu out nodes in graph.
     * @param push_queue Enqueue function.
     * @param push_ret Record result function.
     */
    void SplitTopoOrder(NodePtr& node, const std::set<NodePtr>& nodes, const std::set<NodePtr>& graph_out_set,
                        const std::function<void(std::queue<NodePtr>&, NodePtr&)>& push_queue,
                        const std::function<void(std::set<NodePtr>&)>& push_ret);

    /**
     * @brief Split a graph to reduce circle.
     *
     * @param nodes Graph
     * @return Graphs
     */
    std::vector<std::set<NodePtr>> SplitInternal(std::set<NodePtr>& nodes);

    /**
     * @brief Update node info according to args and connection relation.
     *
     * @param graph Graph nodes.
     * @param graph_num New simple graph id.
     * @param device New device for graph.
     */
    void UpdateInfo(std::set<NodePtr>& graph, int graph_num, const std::string& device);
    /**
     * @brief Bfs to ensure whether previous nodes for a node contain graph out.
     *
     * @param node Node.
     * @return Whether previous nodes for a node contain graph out.
     */
    bool BfsSearchPrivousGraphOut(const std::string& node);
    /**
     * @brief Bfs to ensure whether next nodes for a node contain graph n.
     *
     * @param node Node.
     * @return Whether next nodes for a node contain graph in.
     */
    bool BfsSearchNextGraphIn(const std::string& node);
    /**
     * @brief Build a simple graph which only contain id relation.
     */
    std::pair<SimpleGraph, SimpleGraph> BuildSimpleGraph();

    /**
     * @brief Derive graph out node in level order.
     *
     * @param nodes Graph.
     * @param graph_in_nodes Nodes for topo start.
     * @param graph_out_set Graph out nodes.
     */
    std::vector<NodePtr> TopoForwardGraphOut(std::set<NodePtr>& nodes, std::vector<NodePtr>& graph_in_nodes,
                                             std::set<NodePtr>& graph_out_set);

    /**
     * @brief Setup node info
     *
     * @param graph Whole graph.
     * @param node Node.
     */
    void SetupNode(Graph& graph, const NodePtr& node);
    /**
     * @brief Setup node input info between current node and previous node. Used by SetupNode.
     *
     * @param node Current node
     * @param pre_node Previous Node
     * @param info Record info.
     */
    static void SetupNodeInput(const NodePtr& node, const NodePtr& pre_node, SplitNodeInfo* info);
    /**
     * @brief Setup node output info between current node and next node. Used by SetupNode.
     *
     * @param node Current node
     * @param pre_node Next Node
     * @param info Record info.
     */
    static void SetupNodeOutput(const NodePtr& node, const NodePtr& next_node, SplitNodeInfo* info);
    /**
     * @brief Find all circle.
     */
    std::vector<std::vector<int>> AllCircle();
    /**
     * @brief Derive graph connection.
     *
     * @param graph Graph
     * @param sub_graphs Record connection.
     */
    void DeriveGraphConnection(SubGraphPtr& graph, std::map<int, SubGraphPtr>* sub_graphs);
    /**
     * @brief Derive node input connection. Used by DeriveGraphConnection.
     *
     * @param info Node info.
     * @param subgraph_op_input Record connection.
     */
    void DeriveNodeInputConnection(SplitNodeInfo& info,
                                   std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_input);
    /**
     * @brief Derive node output connection. Used by DeriveGraphConnection.
     *
     * @param info Node info.
     * @param subgraph_op_input Record connection.
     */
    void DeriveNodeOutputConnection(SplitNodeInfo& info,
                                    std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_output);
    /**
     * @brief Derive connection between current node and previous node. Used by DeriveNodeInputConnection.
     *
     * @param info Current node info.
     * @param pre_info Previous node info.
     * @param subgraph_op_input Record connection.
     */
    static void DeriveInputPortConnection(
        SplitNodeInfo& info, SplitNodeInfo& pre_info,
        std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_input);
    /**
     * @brief Derive connection between current node and next node. Used by DeriveNodeInputConnection.
     *
     * @param info Current node info.
     * @param next_info Previous node info.
     * @param subgraph_op_input Record connection.
     */
    static void DeriveOutPortConnection(
        SplitNodeInfo& info, SplitNodeInfo& next_info,
        std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_output);

  public:
    DivideGraphHelper() = default;

    /**
     * @brief DivideGraphHelper merge option
     *
     * @param merge_enable
     */
    explicit DivideGraphHelper(bool merge_enable) : merge_enable(merge_enable) {}

    /**
     * @brief DivideGraphHelper merge option
     *
     * @param merge_enable
     * @param merge_threshold
     */
    DivideGraphHelper(bool merge_enable, uint64_t merge_threshold)
        : merge_enable(merge_enable), merge_threshold(merge_threshold) {}

    /**
     * @brief Setup a graph.
     *
     * @param graph Whole graph.
     */
    void Setup(Graph& graph);
    /**
     * @brief Merge graphs that satisfy condition.
     *
     * 1. The outdegree of graph is 1 and graph size less than a given value.
     * 2. The outdegree of target graph is not 1 (for prevent recursive merge)
     * or The indegree of current graph is 0.
     */
    void Merge();
    /**
     * @brief Merge graphs many times until merge action does not happen.
     */
    void MergeManyTimes();
    /**
     * @brief Split graphs to fix circle.
     */
    void Split();
    /**
     * @brief Combine merge and split to fix circle.
     */
    void FixCircle();
    /**
     * @brief Build final graphs.
     */
    std::map<int, SubGraphPtr> Build();
};

/**
 * @brief Entry for divide graph module.
 *
 * @param graph Graph.
 */
cpp::result<std::map<int, SubGraphPtr>, Error> DivideGraph(Graph& graph);

/**
 * @brief Dfs Search circle in simple DAG.
 *
 * @param simple_graph Graph.
 * @param start Start graph id. Equal to target if begin.
 * @param target Target graph id. Equal to start if begin.
 * @param record Record all circles.
 * @param current Current search state.
 * @param visited A graph is visited.
 */
void DfsForCircle(std::pair<SimpleGraph, SimpleGraph>& simple_graph, int start, int target,
                  std::vector<std::vector<int>>* record, std::vector<int>* current, std::set<int>* visited);

}  // namespace geesibling

#endif
