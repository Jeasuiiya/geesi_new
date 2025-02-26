
#include "DistributedIR/dividegraph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <unordered_set>

#include "DistributedIR/block.hpp"
#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "common/log.h"
#include "range/v3/iterator/concepts.hpp"
#include "range/v3/view/filter.hpp"
#include "range/v3/view/transform.hpp"

namespace geesibling {

bool DivideGraphHelper::BfsSearchPrivousGraphOut(const std::string& before_node) {
    std::queue<std::string> before_node_queue;
    before_node_queue.push(before_node);
    while (!before_node_queue.empty()) {
        std::string the_node = before_node_queue.front();
        SplitNodeInfo& the_node_value = node_infos.find(the_node)->second;
        if (the_node_value.graph_out) {
            return true;
        }
        if (!the_node_value.node->Inputs().empty()) {
            for (const auto& before_node : the_node_value.same_graph_inputs) {
                before_node_queue.push(before_node);
            }
        }
        before_node_queue.pop();
    }
    return false;
}
bool DivideGraphHelper::BfsSearchNextGraphIn(const std::string& next_node) {
    std::queue<std::string> next_node_queue;
    next_node_queue.push(next_node);
    while (!next_node_queue.empty()) {
        std::string the_node = next_node_queue.front();
        SplitNodeInfo& the_node_value = node_infos.find(the_node)->second;
        if (the_node_value.graph_in) {
            return true;
        }
        if (!the_node_value.node->Outputs().empty()) {
            for (const auto& next_node : the_node_value.same_graph_outputs) {
                next_node_queue.push(next_node);
            }
        }
        next_node_queue.pop();
    }
    return false;
}

void DivideGraphHelper::DeriveInputPortConnection(
    SplitNodeInfo& info, SplitNodeInfo& pre_info,
    std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_input) {
    auto& current_node = info.node;
    auto& before_node_real = pre_info.node;
    auto& before_outputs_data = before_node_real->OutputPorts();
    SPDLOG_TRACE("previous node:{} device:{}", before_node_real->Name(), before_node_real->Device());
    if (pre_info.subgraph_num == info.subgraph_num) {
        SPDLOG_ERROR("graph input by self!");
    }
    for (auto& input_data : current_node->InputPorts()) {
        auto iter =
            std::find_if(before_outputs_data.begin(), before_outputs_data.end(), [&](EdgePort<AbstractTensor>& i) {
                return before_node_real->OutputName(i.index) == input_data.entity.Ref();
            });
        // detect tensor connect
        if (iter != before_outputs_data.end()) {
            // 放进map1<input_data, currentnode.name:input_index>
            //  上个节点的第几个输出  当前节点的第几个输入
            auto r = current_node->InputName(input_data.index);
            assert(r.has_value());
            auto data2data = subgraph_op_input->find(pre_info.subgraph_num);
            // 放进subgraph_op_input<前序node所在子图序号，tensor connect>
            if (data2data == subgraph_op_input->end()) {
                std::vector<std::pair<StrAndInt, StrAndInt>> m;
                m.emplace_back(input_data.entity.Ref(), r.value());
                subgraph_op_input->insert({pre_info.subgraph_num, m});
            } else {
                data2data->second.emplace_back(input_data.entity.Ref(), r.value());
            }
        }
    }
}
void DivideGraphHelper::DeriveNodeInputConnection(
    SplitNodeInfo& info, std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_input) {
    auto& current_node = info.node;
    auto& same_graph_inputs = info.same_graph_inputs;
    auto inputs = std::set(current_node->Inputs().begin(), current_node->Inputs().end());
    std::vector<std::string> inputs_diff;
    std::set_difference(inputs.begin(), inputs.end(), same_graph_inputs.begin(), same_graph_inputs.end(),
                        inserter(inputs_diff, inputs_diff.begin()));  // old-->new需要删除的
    for (auto& before_node : inputs_diff) {
        auto before_node_value = node_infos.find(before_node)->second;
        auto& before_node_real = before_node_value.node;
        SPDLOG_TRACE("previous node:{} device:{}", before_node, before_node_real->Device());
        if (before_node_value.subgraph_num == info.subgraph_num) {
            SPDLOG_ERROR("graph input by self!");
        }
        DeriveInputPortConnection(info, before_node_value, subgraph_op_input);
    }
}

void DivideGraphHelper::DeriveOutPortConnection(
    SplitNodeInfo& info, SplitNodeInfo& next_info,
    std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_output) {
    auto& current_node = info.node;
    auto& next_node_real = next_info.node;
    auto next_inputs_data = next_node_real->InputPorts();
    for (auto& output_data : current_node->OutputPorts()) {
        auto iter = std::find_if(next_inputs_data.begin(), next_inputs_data.end(), [&](EdgePort<InputStr>& i) {
            return current_node->OutputName(output_data.index) == i.entity.Ref();
        });
        // detect tensor connect
        if (iter != next_inputs_data.end()) {
            // 这个节点的哪个输出 下个节点的第几个输入
            auto r = next_node_real->InputName(iter->index);
            assert(r.has_value());
            auto data2data = subgraph_op_output->find(next_info.subgraph_num);
            // 放进subgraph_op_input<前序node所在子图序号，tensor connect>
            if (data2data == subgraph_op_output->end()) {
                std::vector<std::pair<StrAndInt, StrAndInt>> m;
                m.emplace_back(iter->entity.Ref(), r.value());
                subgraph_op_output->insert({next_info.subgraph_num, m});
            } else {
                data2data->second.emplace_back(iter->entity.Ref(), r.value());
            }
        }
    }
}
void DivideGraphHelper::DeriveNodeOutputConnection(
    SplitNodeInfo& info, std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>* subgraph_op_output) {
    auto& current_node = info.node;

    auto& same_graph_outputs = info.same_graph_outputs;
    auto outputs = std::set(current_node->Outputs().begin(), current_node->Outputs().end());
    std::vector<std::string> outputs_diff;  // 差集及为前驱后继不同子图的算子
    std::set_difference(outputs.begin(), outputs.end(), same_graph_outputs.begin(), same_graph_outputs.end(),
                        inserter(outputs_diff, outputs_diff.begin()));  // old-->new需要删除的
    for (auto& next_node : outputs_diff) {
        auto next_node_value = node_infos.find(next_node)->second;
        auto& next_node_real = next_node_value.node;
        auto next_inputs_data = next_node_real->InputPorts();
        SPDLOG_TRACE("next node:{} device:{}", next_node, next_node_real->Device());
        if (next_node_value.subgraph_num == info.subgraph_num) {
            SPDLOG_ERROR("graph output by self!");
        }
        // find next node's inputs index
        DeriveOutPortConnection(info, next_node_value, subgraph_op_output);
    }
}
void DivideGraphHelper::DeriveGraphConnection(SubGraphPtr& current_sub_graph, std::map<int, SubGraphPtr>* sub_graphs) {
    // std::map<int, SubGraphPtr> sub_graphs;  // 所有的子图合集
    auto& current_nodes = current_sub_graph->Nodes();
    // map<前驱的子图信息int 对应的节点输入map<string string>>
    std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>> subgraph_op_input;  // before_node 前驱节点中第几个输出
    std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>> subgraph_op_output;  // next_node 后继节点中第几个输入
    // 获得前子图，获得后子图
    // 获得图input   格式：  before节点名:输出index
    // 获得图output  格式：  current节点名:输入index  input_data排序得出

    for (auto& i : current_nodes)  // 遍历子图中的所有节点
    {
        auto& current_node = i;
        SplitNodeInfo& current_node_value = node_infos.find(current_node->Name())->second;
        DeriveNodeInputConnection(current_node_value, &subgraph_op_input);
        DeriveNodeOutputConnection(current_node_value, &subgraph_op_output);
    }
    for (auto& before : subgraph_op_input) {
        int before_subgraph_num = before.first;
        auto& op_op = before.second;
        // auto& before_subgraph = sub_graphs->find(before_subgraph_num)->second;
        auto& before_subgraph = (*sub_graphs)[before_subgraph_num];
        current_sub_graph->AddInputGraph(before_subgraph);
        current_sub_graph->AddInput(op_op);
    }

    for (auto& next : subgraph_op_output) {
        int next_subgraph_num = next.first;
        auto& op_op = next.second;
        // auto& next_subgraph = sub_graphs->find(next_subgraph_num)->second;
        auto& next_subgraph = (*sub_graphs)[next_subgraph_num];
        current_sub_graph->AddOutputGraph(next_subgraph);
        current_sub_graph->AddOutput(op_op);
    }
}
std::map<int, SubGraphPtr> DivideGraphHelper::Build() {
    std::map<int, SubGraphPtr> sub_graphs;  // 所有的子图合集
    // 将算子放进对应的子图
    SPDLOG_DEBUG("subgraph size: {}", graphs.size());
    for (auto& index_nodes : graphs) {
        SPDLOG_TRACE("subgraph {} node size: {}", index_nodes.first, index_nodes.second.size());

        SubGraph sub_graph;
        for (const auto& i : index_nodes.second) {
            sub_graph.AddNode(i);
        }
        sub_graphs.insert({index_nodes.first, std::make_shared<SubGraph>(std::move(sub_graph))});
    }

    // //获取子图连接信息
    for (auto& iter : sub_graphs) {
        auto& current_sub_graph = iter.second;
        DeriveGraphConnection(current_sub_graph, &sub_graphs);
    }
    // 输出子图信息
    for (auto& iter : sub_graphs) {
        std::stringstream ss;
        ss << "======================================" << std::endl;
        int subgraph_num = iter.first;
        auto& current_subgraph = iter.second;
        auto nodes_list = current_subgraph->Nodes();
        ss << "The subgraph num is " << subgraph_num << ". This subgraph contains " << nodes_list.size()
           << " nodes.They are :";
        for (auto& i : nodes_list) {
            NodeBase& the_node = *i;
            ss << the_node.Name() << " ";
        }
        ss << std::endl;

        ss << "before_op_output--current_subgraph_input are :" << std::endl;
        for (auto& op_op : current_subgraph->GetInputs()) {
            for (auto& iter : op_op) {
                ss << iter.first << "--" << iter.second << " ";
            }
            ss << std::endl;
        }
        ss << "current_subgraph_output--next_op_input are :" << std::endl;
        for (auto& op_op : current_subgraph->GetOutputs()) {
            for (auto& iter : op_op) {
                ss << iter.first << "--" << iter.second << " ";
            }
            ss << std::endl;
        }
        SPDLOG_TRACE(ss.str());
    }
    return sub_graphs;
}

std::vector<NodePtr> DivideGraphHelper::TopoForwardGraphOut(std::set<NodePtr>& nodes,
                                                            std::vector<NodePtr>& graph_in_nodes,
                                                            std::set<NodePtr>& graph_out_set) {
    std::queue<std::pair<NodePtr, int>> q;
    std::unordered_set<NodePtr> visited;
    auto push_queue = [&](auto& q, std::pair<NodePtr, int> pair) {
        if (!visited.count(pair.first)) {
            q.push(pair);
            visited.insert(pair.first);
        }
    };
    for (auto& i : graph_in_nodes) {
        push_queue(q, {i, 0});
    }

    // (node , level from start)
    std::vector<std::pair<NodePtr, int>> ret_level;
    while (!q.empty()) {
        auto& pair = q.front();
        auto& node = pair.first;
        q.pop();
        if (graph_out_set.count(node)) {
            ret_level.push_back(pair);
        }

        auto successor = node->Outputs();
        for (auto& i : successor) {
            auto n = node_infos[i].node;
            if (nodes.count(n)) {
                push_queue(q, {n, pair.second + 1});
            }
        }
    }
    // sort by level
    ranges::sort(ret_level, [](auto& a, auto& b) { return a.second < b.second; });
    SPDLOG_TRACE("TopoForwardGraphOut . {}", ret_level);

    auto ret = ret_level | ranges::views::transform([](auto& a) { return a.first; }) | ranges::to_vector;
    if (ret_level.size() != graph_out_set.size()) {
        SPDLOG_ERROR("device out num error. real: {} found: {}", graph_out_set.size(), ret_level.size());
    }
    return ret;
}

void DivideGraphHelper::SplitTopoOrder(NodePtr& node, const std::set<NodePtr>& nodes,
                                       const std::set<NodePtr>& graph_out_set,
                                       const std::function<void(std::queue<NodePtr>&, NodePtr&)>& push_queue,
                                       const std::function<void(std::set<NodePtr>&)>& push_ret) {
    std::queue<NodePtr> q;
    std::set<NodePtr> split_set;
    push_queue(q, node);

    while (!q.empty()) {
        auto& node = q.front();
        q.pop();
        split_set.insert(node);
        // encount first graph out node
        if (graph_out_set.count(node)) {
            push_ret(split_set);
            split_set.clear();
        }
        auto previous = node->Inputs();
        for (auto& i : previous) {
            auto n = node_infos[i].node;
            if (nodes.count(n)) {
                push_queue(q, n);
            }
        }
    }
    push_ret(split_set);
}
std::vector<std::set<NodePtr>> DivideGraphHelper::SplitInternal(std::set<NodePtr>& nodes) {
    std::vector<std::set<NodePtr>> ret;
    auto out_filter = [&](auto& n) {
        return node_infos[n->Name()].graph_out;
    };
    auto in_filter = [&](auto& n) {
        return node_infos[n->Name()].graph_in || n->InputPorts().empty();
    };
    auto push_ret = [&](auto& r) {
        if (!r.empty()) {
            ret.push_back(r);
        }
    };
    std::set<NodePtr> visited;
    auto push_queue = [&](auto& q, auto& node) {
        if (!visited.count(node)) {
            q.push(node);
            visited.insert(node);
        }
    };

    auto graph_in_nodes = nodes | ranges::views::filter(in_filter) | ranges::to_vector;
    auto graph_out_nodes = nodes | ranges::views::filter(out_filter) | ranges::to_vector;
    std::set<NodePtr> graph_out_set(graph_out_nodes.begin(), graph_out_nodes.end());
    auto graph_out_topo_order = TopoForwardGraphOut(nodes, graph_in_nodes, graph_out_set);
    SPDLOG_TRACE("split: graph out nodes {}", graph_out_topo_order);

    for (auto& out_node : graph_out_topo_order) {
        SplitTopoOrder(out_node, nodes, graph_out_set, push_queue, push_ret);
    }
    std::set<NodePtr> rest;
    std::set_difference(nodes.begin(), nodes.end(), visited.begin(), visited.end(), std::inserter(rest, rest.begin()));
    push_ret(rest);

    auto size = std::accumulate(ret.begin(), ret.end(), 0U, [](auto& a, auto& b) { return a + b.size(); });
    if (size != nodes.size()) {
        std::vector<std::string> not_current_graph;
        for (auto& i : ret) {
            for (const auto& n : i) {
                if (nodes.count(n) == 0) {
                    not_current_graph.push_back(n->Name());
                }
            }
            not_current_graph.emplace_back("|");
        }
        SPDLOG_ERROR("split graph: topo error! origin:{}, visited: {}, result: {}, not current graph: {}", nodes.size(),
                     visited.size(), size, not_current_graph);
    }
    SPDLOG_TRACE("finish split graph. origin:{}, visited: {}, nodes to {} graphs", nodes.size(), visited.size(),
                 ret.size());
    return ret;
}

void DivideGraphHelper::UpdateInfo(std::set<NodePtr>& graph, int graph_num, const std::string& device) {
    for (const auto& node : graph) {
        // SPDLOG_TRACE("merge internal: target node: {}", nodes->Name());
        node->Device(device);
        auto& info = node_infos[node->Name()];
        info.subgraph_num = graph_num;

        bool in_other = false;
        info.same_graph_inputs.clear();
        for (auto& input_name : node->Inputs()) {
            auto& input_node = node_infos[input_name].node;
            if (graph.count((input_node))) {
                info.same_graph_inputs.insert(input_name);
            } else {
                in_other = true;
            }
        }
        info.graph_in = in_other;

        bool out_other = false;
        info.same_graph_outputs.clear();
        for (auto& output_name : node->Outputs()) {
            auto& output_node = node_infos[output_name].node;
            if (graph.count((output_node))) {
                info.same_graph_outputs.insert(output_name);
            } else {
                out_other = true;
            }
        }
        info.graph_out = out_other;
    }
}
void DivideGraphHelper::MergeInternal(std::set<NodePtr>& source, int target_num, std::set<NodePtr>* target) {
    if (source.empty() && target->empty()) {
        return;
    }
    std::string device;
    if (!target->empty()) {
        device = (*target->begin())->Device();
    } else {
        device = (*source.begin())->Device();
    }
    target->merge(source);
    UpdateInfo(*target, target_num, device);
}
void DivideGraphHelper::Merge() {
    auto size = graphs.size();
    SPDLOG_DEBUG("merge start: graphs: {}", size);
    auto simple_graph = BuildSimpleGraph();
    // [source -> target]
    std::vector<std::pair<int, int>> merge_schedules;
    for (auto& it : simple_graph.second) {
        auto current_outdegree = it.second.size();
        auto current_graph_size = graphs[it.first].size();
        // outdegree is 1 and nodes < 100 will merge
        if (current_outdegree == 1 && current_graph_size < merge_threshold) {
            auto target = *it.second.begin();
            auto target_out_degree = simple_graph.second[target].size();
            auto current_in_degree_empty = simple_graph.first[it.first].empty();
            // outdegree of target is not 1 to prevent recursive merge
            if ((target_out_degree != 1 || current_in_degree_empty) /*&& it.first != target*/) {
                merge_schedules.emplace_back(it.first, target);
            }
        }
    }
    for (auto& schedule : merge_schedules) {
        SPDLOG_TRACE("merge internal start: {} -> {}", schedule.first, schedule.second);
        MergeInternal(graphs[schedule.first], schedule.second, &graphs[schedule.second]);
        SPDLOG_TRACE("merge internal finish: {} -> {}", schedule.first, schedule.second);
        graphs.erase(schedule.first);
    }
    SPDLOG_DEBUG("merge finish: graphs: {} -> {}", size, graphs.size());
}

void DivideGraphHelper::MergeManyTimes() {
    auto size = graphs.size();
    while (true) {
        Merge();
        if (graphs.size() == size) {
            break;
        }
        size = graphs.size();
    }
}
std::vector<std::vector<int>> DivideGraphHelper::AllCircle() {
    auto simple_graph = BuildSimpleGraph();
    std::vector<std::vector<int>> circles;
    for (size_t i = 0; i < graphs.size(); i++) {
        std::vector<int> current;
        std::set<int> visited;
        DfsForCircle(simple_graph, i, i, &circles, &current, &visited);
        SPDLOG_TRACE("found circle for graph simple id: {}", i);
    }
    return circles;
}
void DivideGraphHelper::Split() {
    auto origin_graph_size = graphs.size();
    auto origin_node_size =
        std::accumulate(graphs.begin(), graphs.end(), 0, [](auto& a, auto& b) { return a + b.second.size(); });
    SPDLOG_DEBUG("fix circle start {} graphs, {} nodes", origin_graph_size, origin_node_size);

    // detect circles
    std::vector<std::vector<int>> circles = AllCircle();
    std::set<int> circle_graphs;
    std::map<int, std::set<NodePtr>> split_subgraphs;
    for (auto& v : circles) {
        for (auto& i : v) {
            circle_graphs.insert(i);
            split_subgraphs[i] = graphs[i];
        }
    }
    // graph in circle is need for spliting, remove
    for (const auto& i : circle_graphs) {
        graphs.erase(i);
    }
    SPDLOG_DEBUG("circle ids: {}", circles);
    SPDLOG_DEBUG("circle_graphs: {}", circle_graphs);

    std::vector<std::set<NodePtr>> split_graphs;
    for (auto& i : split_subgraphs) {
        auto& nodes = i.second;
        auto s = SplitInternal(nodes);
        for (auto& g : s) {
            split_graphs.push_back(g);
        }
    }
    // process splited graphs (origin in circle), assign new simple graph id and update info
    std::map<int, std::set<NodePtr>> ret;
    for (auto& g : split_graphs) {
        auto size = ret.size();
        ret[size] = g;
        if (!g.empty()) {
            UpdateInfo(g, size, (*g.begin())->Device());
        }
        SPDLOG_TRACE("add graph: id:{} node number {} | nodes: {}", size, g.size(), g);
    }
    // process rest graphs (not in circle), assign new simple graph id
    for (auto& g : graphs) {
        auto size = ret.size();
        ret[size] = g.second;
        for (const auto& node : ret[size]) {
            auto& node_value = node_infos[node->Name()];
            node_value.subgraph_num = size;
        }
    }
    SPDLOG_DEBUG("fix circle finish. graph: {} -> {}, node: {} -> {}", origin_graph_size, ret.size(), origin_node_size,
                 std::accumulate(ret.begin(), ret.end(), 0, [](auto& a, auto& b) { return a + b.second.size(); }));
    graphs.swap(ret);
}

std::pair<SimpleGraph, SimpleGraph> DivideGraphHelper::BuildSimpleGraph() {
    SimpleGraph indegree;
    SimpleGraph outdegree;
    for (auto& info_kv : node_infos) {
        // auto gid = info_kv.first;
        auto& info = info_kv.second;
        auto gid = info.subgraph_num;
        auto& node = info.node;

        auto in_it = indegree.find(gid);
        auto input_ids = node->Inputs() | ranges::views::transform([&](auto& s) { return node_infos[s].subgraph_num; })
                         | ranges::to_vector;
        auto in_set = std::set(input_ids.begin(), input_ids.end());
        in_set.erase(gid);
        if (in_it == indegree.end()) {
            indegree.insert({gid, in_set});
        } else {
            indegree[gid].merge(in_set);
        }
        auto out_it = outdegree.find(gid);

        auto output_ids = node->Outputs()
                          | ranges::views::transform([&](auto& s) { return node_infos[s].subgraph_num; })
                          | ranges::to_vector;
        auto out_set = std::set(output_ids.begin(), output_ids.end());
        out_set.erase(gid);
        if (out_it == outdegree.end()) {
            outdegree.insert({gid, out_set});
        } else {
            outdegree[gid].merge(out_set);
        }
    }
    return {indegree, outdegree};
}
void DivideGraphHelper::SetupNodeInput(const NodePtr& node, const NodePtr& pre_node, SplitNodeInfo* info) {
    if (node->Device() == pre_node->Device())  // 同设备的算子记录一下
    {
        SPDLOG_TRACE("same device input: {} {}", node->Name(), pre_node->Name());
        info->same_graph_inputs.insert(pre_node->Name());
    } else {
        info->graph_in = true;
        SPDLOG_TRACE("graph out: {}", pre_node->Name());
    }
}
void DivideGraphHelper::SetupNodeOutput(const NodePtr& node, const NodePtr& next_node, SplitNodeInfo* info) {
    if (node->Device() == next_node->Device()) {
        SPDLOG_TRACE("same device output: {} {}", node->Name(), next_node->Name());
        info->same_graph_outputs.insert(next_node->Name());
    } else {
        info->graph_out = true;
        SPDLOG_TRACE("graph in: {}", next_node->Name());
    }
}
void DivideGraphHelper::SetupNode(Graph& graph, const NodePtr& node) {
    SplitNodeInfo info;
    info.node = node;

    std::string device = node->Device();
    for (auto& input : node->Inputs()) {
        auto& before_node = graph.GetNode(input).value();
        SetupNodeInput(node, before_node, &info);
    }
    for (auto& output : node->Outputs()) {
        auto& next_node = graph.GetNode(output).value();
        SetupNodeOutput(node, next_node, &info);
    }
    node_infos.insert({node->Name(), info});
}
void DivideGraphHelper::Setup(Graph& graph) {
    int node_num = graph.NodeMap().size();
    SPDLOG_DEBUG("node_num: {}", node_num);
    for (auto& node : graph.Nodes()) {
        SetupNode(graph, node);
    }
    AssignGraph();
}

void DivideGraphHelper::AssignGraph() {
    int subgraph_num = -1;
    // 给算子子图编号
    for (auto& iter : node_infos) {
        std::string current_node_name = iter.first;
        auto& current_node_value = iter.second;
        std::queue<std::string> joint_nodes;
        if (current_node_value.subgraph_num == -1) {
            joint_nodes.push(current_node_name);
            subgraph_num++;
            graphs.insert({subgraph_num, {}});
        }
        while (!joint_nodes.empty()) {
            std::string the_node = joint_nodes.front();
            SplitNodeInfo& the_node_value = node_infos.find(the_node)->second;
            the_node_value.subgraph_num = subgraph_num;
            graphs[subgraph_num].insert(the_node_value.node);
            auto& same_graph_inputs = the_node_value.same_graph_inputs;    // 当前算子还相连的前驱算子
            auto& same_graph_outputs = the_node_value.same_graph_outputs;  // 当前算子还相连的后继算子
            for (const auto& in_op : same_graph_inputs) {
                SplitNodeInfo& in_op_value = node_infos.find(in_op)->second;
                if (in_op_value.subgraph_num == -1) {
                    joint_nodes.push(in_op);
                }
            }
            for (const auto& out_op : same_graph_outputs) {
                SplitNodeInfo& out_op_value = node_infos.find(out_op)->second;
                if (out_op_value.subgraph_num == -1) {
                    joint_nodes.push(out_op);
                }
            }
            joint_nodes.pop();
        }
    }
    for (auto& it : graphs) {
        std::vector<std::string> out;
        std::for_each(it.second.begin(), it.second.end(), [&](auto& i) { out.push_back(i->Name()); });
        SPDLOG_TRACE("init graph {}. nodes: {}", it.first, out);
    }
}
void DivideGraphHelper::FixCircle() {
    auto size = graphs.size();
    while (true) {
        if (merge_enable) {
            MergeManyTimes();
        }
        Split();
        if (size == graphs.size()) {
            break;
        }
        size = graphs.size();
    }
    std::vector<int> zeros;
    for (auto& it : graphs) {
        if (it.second.empty()) {
            zeros.push_back(it.first);
        }
    }
    for (auto& i : zeros) {
        graphs.erase(i);
    }
}
// todo: no recursive dfs
// NOLINTBEGIN
void DfsForCircle(std::pair<SimpleGraph, SimpleGraph>& simple_graph, int start, int target,
                  std::vector<std::vector<int>>* record, std::vector<int>* current, std::set<int>* visited) {
    SimpleGraph& indegree = simple_graph.first;
    current->push_back(start);
    visited->insert(start);
    for (const auto& next : indegree[start]) {
        if (next == target) {
            SPDLOG_TRACE("DfsForCircle found circle {} -> {} -> {}, current: {}", next, start, target, (*current));
            record->push_back(*current);
        } else if (!visited->count(next)) {
            DfsForCircle(simple_graph, next, target, record, current, visited);
        }
    }
    current->pop_back();
}
// NOLINTEND
//
cpp::result<std::map<int, SubGraphPtr>, Error> DivideGraph(Graph& graph) {
    DivideGraphHelper helper;
    helper.Setup(graph);
    helper.FixCircle();
    auto sub_graph = helper.Build();
    SPDLOG_DEBUG("DivideGraph finish. subgraph size: {}", sub_graph.size());
    return sub_graph;
}

}  // namespace geesibling
