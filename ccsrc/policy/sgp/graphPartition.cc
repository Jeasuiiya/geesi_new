#include "policy/sgp/graphPartition.h"

#include "common/log.h"

namespace geesibling {

std::int64_t Partition::GetOpMemory(Graph& graph, const std::string& op) {
    auto node = *graph.GetNode(op);
    return node->OutputMemory() + node->ComputeCost();
}

Partition::Partition(Graph& graph, std::int64_t group_num, std::vector<Device> devices, float target_factor, int OOM) {
    this->builder.ConstructBuilder(graph, OOM);
    std::vector<std::shared_ptr<NodeBase>>& op_graph_ops_dict = this->builder.graph.Nodes();
    std::int64_t graph_total_cost = 0;
    for (auto& node : op_graph_ops_dict) {
        std::map<std::string, std::string> attr = (*node).Attrs();
        std::string group_name = attr["colocation_group"];
        if (group_info.count(group_name) == 0) {
            std::vector<std::string> ops;
            ops.push_back((*node).Name());
            group_info.insert(std::pair<std::string, std::vector<std::string>>(group_name, ops));
        } else {
            group_info[group_name].push_back((*node).Name());
        }
        graph_total_cost += this->GetOpMemory(this->builder.graph, (*node).Name());
    }
    this->group_balance_cost = graph_total_cost / group_num;
    for (std::int64_t i = 0; i < group_num; i++) {
        group_list.emplace_back(i, this->group_balance_cost);
    }
    this->GraphPlacement(this->builder.graph, std::move(devices), 500, target_factor);
}

void Partition::GraphPlacement(Graph& graph, std::vector<Device> devices, std::int64_t max_iterations,
                               float target_factor) {
    std::map<int64_t, std::string> id_device_map;
    for (uint64_t i = 0; i < this->group_list.size(); i++) {
        id_device_map[this->group_list[i].id] = devices[i].GetName();
    }
    this->Split(graph);
    this->AdjustV2(this->builder.graph, max_iterations, target_factor, id_device_map);
    for (auto& it : this->group_list) {
        std::set<std::string> op_member = it.op_member;
        for (const auto& op : op_member) {
            std::string device = id_device_map[it.id];
            this->op_group.insert(std::pair<std::string, std::string>(op, device));
            SPDLOG_DEBUG("Node: {}, Device id: {}", op, device);
        }
    }
}

void Partition::Split(Graph& graph) {
    uint64_t index = 0;
    uint64_t device_num = this->group_list.size();
    std::set<std::string> visited;
    std::vector<std::shared_ptr<NodeBase>>& graph_nodes = graph.Nodes();
    for (auto& node : graph_nodes) {
        std::string op = (*node).Name();
        std::map<std::string, std::int64_t> out_edge = this->builder.out_edge[op];
        for (auto& it : out_edge) {
            if (it.second > 0) {
                std::string str = it.first;
                this->critical_edges.emplace_back(op, str, it.second);
            }
        }
        if (visited.count(op)) {
            continue;
        }
        if (this->group_list[index].GetTotalCost() >= this->group_balance_cost) {
            if (index != this->group_list.size() - 1) {
                index = (index + 1) % device_num;
            }
        }
        std::map<std::string, std::string> attr = (*node).Attrs();
        std::string group_name = attr["colocation_group"];
        std::vector<std::string> mem_ops = this->group_info[group_name];
        for (auto& mem_op : mem_ops) {
            this->group_list[index].AddOp(mem_op);
            this->op_group_dict[mem_op] = index;
            visited.insert(mem_op);
            this->group_list[index].comm_cost_inner += this->GetOpMemory(this->builder.graph, mem_op);
        }
    }
}

bool Partition::Cmp(const SgpEdge& e1, const SgpEdge& e2) {
    return e1.comm_cost > e2.comm_cost;
}

void Partition::AdjustV2(Graph& graph, std::int64_t max_iterations, float target_factor,
                         std::map<int64_t, std::string> id_device_map) {
    for (auto& edge : this->critical_edges) {
        std::string pre = edge.pre_op;
        std::string succ = edge.succ_op;
        std::int64_t pre_group_id = this->op_group_dict[pre];
        std::int64_t succ_group_id = this->op_group_dict[succ];
        std::string device_pre = id_device_map[pre_group_id];
        std::string device_succ = id_device_map[succ_group_id];
        std::string task1 = device_pre.substr(device_pre.find("task:") + 5);
        std::string task2 = device_succ.substr(device_succ.find("task:") + 5);
        if (task1 != task2) {
            edge.comm_cost *= 100;
        }
    }
    for (int i = 0; i < max_iterations; i++) {
        sort(this->critical_edges.begin(), this->critical_edges.end(), Cmp);
        for (auto edge : this->critical_edges) {
            std::string pre = edge.pre_op;
            std::string succ = edge.succ_op;
            std::map<std::string, std::string> attr;
            attr = (*graph.GetNode(pre))->Attrs();
            std::string pre_group_name = attr["colocation_group"];
            attr = (*graph.GetNode(succ))->Attrs();
            std::string succ_group_name = attr["colocation_group"];
            std::int64_t pre_group_id = this->op_group_dict[pre];
            std::int64_t succ_group_id = this->op_group_dict[succ];
            if (pre_group_id != succ_group_id) {
                std::int64_t comm_cost_outer = 0;
                if (this->builder.out_edge.count(succ)) {
                    std::map<std::string, std::int64_t> out_edge = this->builder.out_edge[succ];
                    for (auto& it : out_edge) {
                        if (pre_group_id != this->op_group_dict[it.first]) {
                            comm_cost_outer += it.second;
                        }
                    }
                }
                if (edge.comm_cost >= comm_cost_outer
                    && this->group_list[pre_group_id].balance_factor <= target_factor) {
                    if (this->group_info.count(succ_group_name)) {
                        std::vector<std::string> ops = this->group_info.at(succ_group_name);
                        for (auto& op : ops) {
                            this->group_list[pre_group_id].op_member.insert(op);
                            this->group_list[pre_group_id].comm_cost_inner += this->GetOpMemory(graph, op);
                            this->group_list[succ_group_id].op_member.erase(op);
                            this->group_list[succ_group_id].comm_cost_inner -= this->GetOpMemory(graph, op);
                            this->op_group_dict[op] = pre_group_id;
                        }
                    }
                } else {
                    comm_cost_outer = 0;
                    if (this->builder.in_edge.count(pre)) {
                        std::map<std::string, std::int64_t> in_edge = this->builder.in_edge[pre];
                        for (auto& it : in_edge) {
                            if (succ_group_id != this->op_group_dict[it.first]) {
                                comm_cost_outer += it.second;
                            }
                        }
                    }
                    if (edge.comm_cost >= comm_cost_outer
                        && this->group_list[succ_group_id].balance_factor <= target_factor) {
                        if (this->group_info.count(pre_group_name)) {
                            std::vector<std::string> ops = this->group_info[pre_group_name];
                            for (auto& op : ops) {
                                this->group_list[succ_group_id].op_member.insert(op);
                                this->group_list[succ_group_id].comm_cost_inner += this->GetOpMemory(graph, op);
                                this->group_list[pre_group_id].op_member.erase(op);
                                this->group_list[pre_group_id].comm_cost_inner -= this->GetOpMemory(graph, op);
                                this->op_group_dict[op] = succ_group_id;
                            }
                        }
                    }
                }
                this->group_list[pre_group_id].UpdateBalanceFactor();
                this->group_list[succ_group_id].UpdateBalanceFactor();
                edge.comm_cost = edge.comm_cost * (0.5) * (1 + cos(((1 - 1.0 / (edge.choosed_times + 1)) * M_PI)));
                edge.choosed_times += 1;
            }
        }
    }
}

}  // namespace geesibling
