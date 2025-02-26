#include "policy/fd-dps/fddps_algorithm.h"

#include "range/v3/all.hpp"
namespace geesibling {

cpp::result<std::vector<CostNode>, Error> FDDPSAlgorithm::Placement() {
    std::vector<CostNode>& cost_nodes = cost_graph.GetCostNodes();
    // 判断所有设备的总内存是否满足调度要求, 不满足就退出
    int64_t all_memory = 0;
    int64_t limit_memory = 0;
    for (auto& device : devices) {
        limit_memory += device.GetFreeMemory();
    }
    for (auto& cost_node : cost_nodes) {
        all_memory += cost_node.GetMemoryCost();
    }
    if (all_memory > limit_memory) {
        return cpp::fail(Error(Kind::Invalid, fmt::format("{}:{} out of memory.", __FILE__, __LINE__)));
    }

    // 设立一个记录 indegree 的 map, 方便之后的调度
    // 初始化 name_to_indegree 和 name_to_priority
    std::map<std::string, int64_t> name_to_indegree;
    std::map<std::string, double> name_to_priority;
    for (auto& cost_node : cost_nodes) {
        std::string& name = cost_node.GetName();
        int64_t indegree = cost_node.GetInputs().size();
        name_to_indegree.insert(std::pair<std::string, int64_t>(name, indegree));
        if (indegree == 0) {
            name_to_priority.insert(std::pair<std::string, double>(name, 0));
        }
    }

    // 初始化 node_map
    std::map<std::string, CostNode> node_map;
    for (auto& cost_node : cost_nodes) {
        node_map.insert(std::pair<std::string, CostNode>(cost_node.GetName(), cost_node));
    }

    // 设立一个用于更新开始时间和结束时间的 vector, 更新节点开始时间和结束时间按照这个顺序执行
    std::vector<std::string> update_sequence;
    std::string input_name = name_to_priority.begin()->first;
    GetUpdateSequence(input_name, node_map, name_to_indegree, &update_sequence);

    std::map<std::string, int64_t> earlist_start_times;
    std::map<std::string, int64_t> latest_start_times;
    int64_t total_time;
    InitStartAndEndTime(update_sequence, node_map, &earlist_start_times, &latest_start_times, &total_time);

    // 被调度的节点的值设置为1, 未被调度的节点设置为0
    std::map<std::string, int64_t> scheduled_nodes;
    for (auto& node : cost_nodes) {
        scheduled_nodes.insert(std::pair<std::string, int64_t>(node.GetName(), 0));
    }
    // 将 devices 转换成 device_map 方便索引
    std::map<std::string, Device> device_map;
    for (auto& device : devices) {
        device_map.insert(std::pair<std::string, Device>(device.GetName(), device));
    }

    // get max execute time
    int64_t max_execute_time = 0;
    for (auto cost_node : cost_nodes) {
        std::vector<int64_t> output_comm_costs = cost_node.GetOutputCommCosts();
        int64_t all_comm_costs = 0;
        for (auto& output : output_comm_costs) {
            all_comm_costs += output;
        }
        max_execute_time += cost_node.GetComputeCost() + all_comm_costs;
    }

    // node_to_device 代表节点被调度的设备
    std::map<std::string, std::string> node_to_device;
    while (!name_to_priority.empty()) {
        std::string schedule_node_name;

        // 获取需要被调度的节点
        GetScheduleNode(total_time, earlist_start_times, latest_start_times, &name_to_priority, &schedule_node_name);
        scheduled_nodes[schedule_node_name] = 1;
        // 首先找和前驱节点通信开销最大的节点
        CostNode schedule_node = node_map[schedule_node_name];
        std::string best_device;
        int64_t execute_time = max_execute_time;
        TRY(GetBestDevice(schedule_node, node_map, &device_map, &best_device, &execute_time));
        ScheduleNodeToDevice(schedule_node, best_device, execute_time, &node_map, &device_map, &node_to_device);
        UpdateStartAndEndTime(update_sequence, scheduled_nodes, total_time, &node_map, &earlist_start_times,
                              &latest_start_times);
        UpdateScheduleableNodes(schedule_node, &name_to_indegree, &name_to_priority);
    }

    return node_map | ranges::views::values | ranges::to<std::vector<CostNode>>();
}

void FDDPSAlgorithm::GetPriority(int64_t total_time, std::map<std::string, int64_t> earlist_start_times,
                                 std::map<std::string, int64_t> latest_start_times,
                                 std::map<std::string, double>* name_to_priority, std::string* schedule_node_name) {
    int64_t best_priority = total_time;
    for (auto& name_priority : *name_to_priority) {
        int64_t priority = latest_start_times[name_priority.first] - earlist_start_times[name_priority.first];
        name_priority.second = priority;
        if (best_priority >= priority) {
            best_priority = priority;
            *schedule_node_name = name_priority.first;
        }
    }
}

void FDDPSAlgorithm::GetScheduleNode(int64_t total_time, std::map<std::string, int64_t> earlist_start_times,
                                     std::map<std::string, int64_t> latest_start_times,
                                     std::map<std::string, double>* name_to_priority, std::string* schedule_node_name) {
    if (name_to_priority->size() == 1) {
        auto scheduled_node = name_to_priority->begin();
        *schedule_node_name = scheduled_node->first;
    } else {
        GetPriority(total_time, std::move(earlist_start_times), std::move(latest_start_times), name_to_priority,
                    schedule_node_name);
    }
    name_to_priority->erase(*schedule_node_name);
}

void FDDPSAlgorithm::GetUpdateSequence(std::string& input_name, std::map<std::string, CostNode> node_map,
                                       std::map<std::string, int64_t> name_to_indegree,
                                       std::vector<std::string>* update_sequence) {
    std::queue<std::string> node_queues;
    node_queues.push(input_name);
    while (!node_queues.empty()) {
        std::string node = node_queues.front();
        update_sequence->push_back(node);
        node_queues.pop();
        for (auto& output : node_map[node].GetOutputs()) {
            name_to_indegree[output]--;
            if (name_to_indegree[output] == 0) {
                node_queues.push(output);
            }
        }
    }
}

cpp::result<void, Error> FDDPSAlgorithm::GetBestDevice(CostNode schedule_node, std::map<std::string, CostNode> node_map,
                                                       std::map<std::string, Device>* device_map,
                                                       std::string* best_device, int64_t* execute_time) {
    std::vector<std::string>& inputs = schedule_node.GetInputs();

    for (auto& device_pair : *device_map) {
        Device& device = device_pair.second;
        if (device.GetFreeMemory() < schedule_node.GetMemoryCost()) {
            continue;
        }
        int64_t device_execute_time = device.GetExecuteTime();
        int64_t node_end_time = 0;
        std::string& device_name = device.GetName();
        if (inputs.empty()) {
            node_end_time = device_execute_time + schedule_node.GetComputeCost();
        }
        for (unsigned int i = 0; i < inputs.size(); i++) {
            int64_t comm_cost = schedule_node.GetInputCommCosts()[i];
            if (device_name == node_map[inputs[i]].GetDevice()) {
                node_end_time = node_end_time > schedule_node.GetComputeCost() + device_execute_time
                                    ? node_end_time
                                    : schedule_node.GetComputeCost() + device_execute_time;
            } else {
                CostNode& input = node_map[inputs[i]];
                int64_t input_end_time = input.GetEndTime();
                node_end_time = node_end_time > schedule_node.GetComputeCost() + input_end_time + comm_cost
                                    ? node_end_time
                                    : schedule_node.GetComputeCost() + input_end_time + comm_cost;
            }
        }

        if (*execute_time >= node_end_time) {
            *execute_time = node_end_time;
            *best_device = device.GetName();
        }
    }

    if (best_device->empty()) {
        return cpp::fail(
            Error(Kind::Invalid, fmt::format("{}:{} no node statsifies the schedule node {}. out of memory. ", __FILE__,
                                             __LINE__, schedule_node.GetName())));
    }
    return {};
}

void FDDPSAlgorithm::ScheduleNodeToDevice(CostNode schedule_node, std::string& best_device, int64_t execute_time,
                                          std::map<std::string, CostNode>* node_map,
                                          std::map<std::string, Device>* device_map,
                                          std::map<std::string, std::string>* node_to_device) {
    std::string node_name = schedule_node.GetName();
    CostNode& node = (*node_map)[node_name];
    Device& device = (*device_map)[best_device];
    // 更新设备的内存和执行时间, 更新节点的开始时间和结束时间
    node.SetStartTime(execute_time - node.GetComputeCost());
    node.SetEndTime(execute_time);
    node.SetDevice(best_device);
    std::vector<int64_t>& input_comm_costs = node.GetInputCommCosts();
    std::vector<std::string>& inputs = node.GetInputs();
    for (unsigned int i = 0; i < inputs.size(); i++) {
        CostNode& input = (*node_map)[inputs[i]];
        // 更新节点间的通信开销
        if (input.GetDevice() == node.GetDevice()) {
            input_comm_costs[i] = 0;
            std::vector<int64_t>& input_output_comm_costs = input.GetOutputCommCosts();
            std::vector<std::string>& input_outputs = input.GetOutputs();
            for (unsigned int j = 0; j < input_outputs.size(); j++) {
                if (input_outputs[j] == node_name) {
                    input_output_comm_costs[j] = 0;
                    break;
                }
            }
        }
    }
    device.SetExecuteTime(execute_time);
    int64_t free_memory = device.GetFreeMemory();
    device.SetFreeMemory(free_memory - node.GetMemoryCost());
    node_to_device->insert(std::pair<std::string, std::string>(node_name, best_device));
}

void FDDPSAlgorithm::UpdateStartAndEndTime(std::vector<std::string> update_sequence,
                                           std::map<std::string, int64_t> scheduled_nodes, int64_t total_time,
                                           std::map<std::string, CostNode>* node_map,
                                           std::map<std::string, int64_t>* earlist_start_times,
                                           std::map<std::string, int64_t>* latest_start_times) {
    // 更新所有节点的 earlist_start_times, latest_start_times 和 start_time, end_time
    // 确定需要更新的节点
    for (auto& node_name : update_sequence) {
        if (scheduled_nodes[node_name] == 1) {
            continue;
        }
        CostNode& node = (*node_map)[node_name];
        std::vector<std::string> inputs = node.GetInputs();
        std::vector<int64_t> input_comm_costs = node.GetInputCommCosts();
        int64_t max_start_time = 0;
        for (unsigned int i = 0; i < inputs.size(); i++) {
            CostNode& input = (*node_map)[inputs[i]];
            int64_t input_end_time = input.GetEndTime();
            max_start_time = max_start_time > input_end_time + input_comm_costs[i]
                                 ? max_start_time
                                 : input_end_time + input_comm_costs[i];
        }
        node.SetStartTime(max_start_time);
        node.SetEndTime(max_start_time + node.GetComputeCost());
        (*earlist_start_times)[node_name] = max_start_time;
    }

    // 在所有节点更新之后, 更新latest_start_times
    int64_t node_num = update_sequence.size();
    for (int i = node_num - 2; i >= 0; i--) {
        std::string node_name = update_sequence[i];
        if (scheduled_nodes[node_name] == 1) {
            continue;
        }
        CostNode node = (*node_map)[node_name];
        std::vector<int64_t> node_last_start_times;
        std::vector<std::string> outputs = node.GetOutputs();
        std::vector<int64_t> output_comm_costs = node.GetOutputCommCosts();
        int64_t output_num = outputs.size();
        int64_t compute_cost = node.GetComputeCost();
        int64_t min_last_start_time = total_time;
        for (unsigned int j = 0; j < output_num; j++) {
            CostNode& output = (*node_map)[outputs[j]];
            int64_t output_start_time = output.GetStartTime();
            min_last_start_time = min_last_start_time < output_start_time - output_comm_costs[j] - compute_cost
                                      ? min_last_start_time
                                      : output_start_time - output_comm_costs[j] - compute_cost;
        }
        (*latest_start_times)[node_name] = min_last_start_time;
    }
}

void FDDPSAlgorithm::UpdateScheduleableNodes(CostNode schedule_node, std::map<std::string, int64_t>* name_to_indegree,
                                             std::map<std::string, double>* name_to_priority) {
    std::vector<std::string>& outputs = schedule_node.GetOutputs();
    for (auto& output : outputs) {
        (*name_to_indegree)[output]--;
        if ((*name_to_indegree)[output] == 0) {
            name_to_priority->insert(std::pair<std::string, double>(output, 0));
        }
    }
}

void FDDPSAlgorithm::InitStartAndEndTime(std::vector<std::string> update_sequence,
                                         std::map<std::string, CostNode> node_map,
                                         std::map<std::string, int64_t>* earlist_start_times,
                                         std::map<std::string, int64_t>* latest_start_times, int64_t* total_time) {
    int64_t node_num = update_sequence.size();
    std::string& output_name = update_sequence[node_num - 1];
    CostNode& output = node_map[output_name];
    *total_time = output.GetEndTime();
    latest_start_times->insert(std::pair<std::string, int64_t>(output_name, output.GetStartTime()));
    for (unsigned int i = 0; i < node_num; i++) {
        std::string& node_name = update_sequence[i];
        CostNode& node = node_map[node_name];
        earlist_start_times->insert(std::pair<std::string, int64_t>(node_name, node.GetStartTime()));
    }
    for (int i = node_num - 2; i >= 0; i--) {
        std::string& node_name = update_sequence[i];
        CostNode& node = node_map[node_name];
        std::vector<int64_t> node_last_start_times;
        std::vector<std::string>& outputs = node.GetOutputs();
        std::vector<int64_t>& output_comm_costs = node.GetOutputCommCosts();
        int64_t output_num = outputs.size();
        int64_t min_last_start_time = *total_time;
        int64_t compute_cost = node.GetComputeCost();
        for (unsigned int j = 0; j < output_num; j++) {
            min_last_start_time =
                min_last_start_time < (*latest_start_times)[outputs[j]] - output_comm_costs[j] - compute_cost
                    ? min_last_start_time
                    : (*latest_start_times)[outputs[j]] - output_comm_costs[j] - compute_cost;
        }
        latest_start_times->insert(std::pair<std::string, int64_t>(node_name, min_last_start_time));
    }
}
}  // namespace geesibling
