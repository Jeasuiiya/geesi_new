#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "cost_graph/cost_graph.hpp"

#ifndef GEESIBLING_COST_GRAPH_COMMON_HPP
#define GEESIBLING_COST_GRAPH_COMMON_HPP

namespace geesibling {
inline CostGraph ConvertGraphToCostGraph(Graph graph) {
    const std::vector<NodePtr>& graph_nodes = graph.Nodes();

    std::vector<CostNode> cost_nodes;
    std::map<std::string, CostNode&> cost_node_map;
    for (const auto& node : graph_nodes) {
        CostNode cost_node(*node);
        cost_nodes.push_back(cost_node);
        cost_node_map.insert(std::pair<std::string, CostNode&>(cost_node.GetName(), cost_node));
    }

    CostGraph cost_graph(cost_nodes, cost_node_map);
    return cost_graph;
}

inline MergedCostGraph InitMergedCostGraph(CostGraph cost_graph) {
    const std::vector<CostNode>& cost_nodes = cost_graph.GetCostNodes();
    std::vector<MergedCostNode> merged_cost_nodes;
    std::map<std::string, MergedCostNode&> merged_cost_node_map;
    for (const auto& node : cost_nodes) {
        MergedCostNode merged_cost_node(node);
        merged_cost_nodes.push_back(merged_cost_node);
        merged_cost_node_map.insert(
            std::pair<std::string, MergedCostNode&>(merged_cost_node.GetName(), merged_cost_node));
    }

    MergedCostGraph merged_node_graph(merged_cost_nodes, merged_cost_node_map);
    return merged_node_graph;
}

inline CostGraph ConvertMergedCostGraphToCostGraph(MergedCostGraph merged_cost_graph) {
    std::vector<MergedCostNode>& merged_cost_nodes = merged_cost_graph.GetMergedCostNodes();
    std::vector<CostNode> cost_nodes;
    std::map<std::string, CostNode&> cost_node_map;
    for (auto& node : merged_cost_nodes) {
        CostNode cost_node = static_cast<CostNode>(node);
        cost_nodes.push_back(cost_node);
        cost_node_map.insert(std::pair<std::string, CostNode&>(cost_node.GetName(), cost_node));
    }

    CostGraph cost_graph(cost_nodes, cost_node_map);
    return cost_graph;
}
}  // namespace geesibling

#endif
