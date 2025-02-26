#include <map>
#include <string>
#include <vector>

#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "common/util.hpp"

#ifndef GEESIBLING_COST_GRAPH_COST_GRAPH_HPP
#define GEESIBLING_COST_GRAPH_COST_GRAPH_HPP

namespace geesibling {

// Edge
class CostEdge {
  private:
    std::string left_node;
    std::string right_node;
    int64_t comm_cost;

  public:
    CostEdge() : comm_cost(0) {}
    explicit CostEdge(CostEdge* edge)
        : left_node(edge->GetLeftNode()), right_node(edge->GetRightNode()), comm_cost(edge->GetCommCost()){};
    CostEdge(std::string _left_node, std::string _right_node, int64_t _comm_cost)
        : left_node(std::move(_left_node)), right_node(std::move(_right_node)), comm_cost(std::move(_comm_cost)){};

    virtual ~CostEdge() = default;

    DECL_ACCESSOR(GetLeftNode, SetLeftNode, left_node, M)
    DECL_ACCESSOR(GetRightNode, SetRightNode, right_node, M)
    DECL_ACCESSOR(GetCommCost, SetCommCost, comm_cost, M)
};

class CostNode {
  private:
    std::string name;
    std::string device;
    int64_t compute_cost;
    int64_t memory_cost;
    int64_t output_memory;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int64_t> input_comm_costs;
    std::vector<int64_t> output_comm_costs;
    int64_t start_time;
    int64_t end_time;

  public:
    CostNode() : compute_cost(0), memory_cost(0), output_memory(0), start_time(0), end_time(0) {}
    explicit CostNode(NodeBase node)
        : name(std::move(node.Name())),
          device(std::move(node.Device())),
          compute_cost(std::move(node.ComputeCost())),
          memory_cost(std::move(node.PersistentMemory())),
          output_memory(std::move(node.OutputMemory())),
          inputs(std::move(node.Inputs())),
          outputs(std::move(node.Outputs())),
          start_time(std::move(node.StartTime())),
          end_time(std::move(node.EndTime())) {
        int64_t input_mem = node.InputMemory();
        int64_t input_node_num = inputs.size();
        if (input_node_num != 0) {
            input_comm_costs = std::vector<int64_t>(input_node_num, static_cast<int64_t>(input_mem / input_node_num));
        }
        int64_t output_mem = node.OutputMemory();
        int64_t output_node_num = outputs.size();
        if (output_node_num != 0) {
            output_comm_costs =
                std::vector<int64_t>(output_node_num, static_cast<int64_t>(output_mem / output_node_num));
        }
    }
    CostNode(std::string _name, std::string _device, int64_t _compute_cost, int64_t _memory_cost,
             int64_t _output_memory, std::vector<std::string> _inputs, std::vector<std::string> _outputs,
             std::vector<int64_t> _input_comm_costs, std::vector<int64_t> _output_comm_costs, int64_t _start_time,
             int64_t _end_time)
        : name(std::move(_name)),
          device(std::move(_device)),
          compute_cost(std::move(_compute_cost)),
          memory_cost(std::move(_memory_cost)),
          output_memory(std::move(_output_memory)),
          inputs(std::move(_inputs)),
          outputs(std::move(_outputs)),
          input_comm_costs(std::move(_input_comm_costs)),
          output_comm_costs(std::move(_output_comm_costs)),
          start_time(std::move(_start_time)),
          end_time(std::move(_end_time)) {}

    virtual ~CostNode() = default;

    DECL_ACCESSOR(GetName, SetName, name, M)
    DECL_ACCESSOR(GetDevice, SetDevice, device, M)
    DECL_ACCESSOR(GetComputeCost, SetComputeCost, compute_cost, M)
    DECL_ACCESSOR(GetMemoryCost, SetMemoryCost, memory_cost, M)
    DECL_ACCESSOR(GetOutputMemory, SetOutputMemory, memory_cost, M)
    DECL_ACCESSOR(GetInputs, SetInputs, inputs, M)
    DECL_ACCESSOR(GetOutputs, SetOutputs, outputs, M)
    DECL_ACCESSOR(GetInputCommCosts, SetInputCommCosts, input_comm_costs, M)
    DECL_ACCESSOR(GetOutputCommCosts, SetOutputCommCosts, output_comm_costs, M)
    DECL_ACCESSOR(GetStartTime, SetStartTime, start_time, M)
    DECL_ACCESSOR(GetEndTime, SetEndTime, end_time, M)
};

class MergedCostNode : public CostNode {
  private:
    std::vector<CostNode> cost_nodes;
    std::vector<std::string> cost_node_names;

  public:
    MergedCostNode() = default;
    explicit MergedCostNode(CostNode node)
        : CostNode(node.GetName(), node.GetDevice(), node.GetComputeCost(), node.GetMemoryCost(),
                   node.GetOutputMemory(), node.GetInputs(), node.GetOutputs(), node.GetInputCommCosts(),
                   node.GetOutputCommCosts(), node.GetStartTime(), node.GetEndTime()) {
        cost_nodes.push_back(node);
        cost_node_names.push_back(node.GetName());
    }

    DECL_ACCESSOR(GetCostNodes, SetCostNodes, cost_nodes, M)
    DECL_ACCESSOR(GetCostNodeNames, SetCostNodeNames, cost_node_names, M)
};

class MergedCostGraph {
  private:
    std::vector<MergedCostNode> merged_cost_nodes;
    std::map<std::string, MergedCostNode&> merged_cost_node_map;

  public:
    MergedCostGraph() = default;
    MergedCostGraph(std::vector<MergedCostNode> _merged_cost_nodes,
                    std::map<std::string, MergedCostNode&> _merged_cost_node_map)
        : merged_cost_nodes(std::move(_merged_cost_nodes)), merged_cost_node_map(std::move(_merged_cost_node_map)) {}
    virtual ~MergedCostGraph() = default;

    DECL_ACCESSOR(GetMergedCostNodes, SetMergedCostNodes, merged_cost_nodes, M)
    DECL_ACCESSOR(GetMergedCostNodeMap, SetMergedCostNodeMap, merged_cost_node_map, M)
};

class CostGraph {
  private:
    std::vector<CostNode> cost_nodes;
    std::map<std::string, CostNode&> cost_node_map;

  public:
    CostGraph() = default;
    explicit CostGraph(CostGraph* cost_graph)
        : cost_nodes(cost_graph->GetCostNodes()), cost_node_map(cost_graph->GetCostNodeMap()) {}
    CostGraph(std::vector<CostNode> _cost_nodes, std::map<std::string, CostNode&> _cost_node_map)
        : cost_nodes(std::move(_cost_nodes)), cost_node_map(std::move(_cost_node_map)) {}
    virtual ~CostGraph() = default;

    DECL_ACCESSOR(GetCostNodes, SetMergedCostNodes, cost_nodes, M)
    DECL_ACCESSOR(GetCostNodeMap, SetCostNodeMap, cost_node_map, M)
};
}  // namespace geesibling
#endif
