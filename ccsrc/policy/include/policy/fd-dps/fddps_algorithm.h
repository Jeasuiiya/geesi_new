#include <queue>

#include "cluster/server.hpp"
#include "common/error.hpp"
#include "common/result_macro.hpp"
#include "cost_graph/cost_graph.hpp"

#ifndef GEESIBLING_FDDPS_ALGORITHM_H
#define GEESIBLING_FDDPS_ALGORITHM_H

namespace geesibling {
class FDDPSAlgorithm {
  private:
    CostGraph cost_graph;
    std::vector<Device> devices;

  public:
    FDDPSAlgorithm() = default;
    explicit FDDPSAlgorithm(CostGraph& _cost_graph) : cost_graph(std::move(_cost_graph)){};
    FDDPSAlgorithm(CostGraph& _cost_graph, std::vector<Device> _devices)
        : cost_graph(std::move(_cost_graph)), devices(std::move(_devices)){};

    virtual ~FDDPSAlgorithm() = default;

    cpp::result<std::vector<CostNode>, Error> Placement();
    static void GetPriority(int64_t total_time, std::map<std::string, int64_t> earlist_start_times,
                            std::map<std::string, int64_t> latest_start_times,
                            std::map<std::string, double>* name_to_priority, std::string* schedule_node_name);
    static void GetScheduleNode(int64_t total_time, std::map<std::string, int64_t> earlist_start_times,
                                std::map<std::string, int64_t> latest_start_times,
                                std::map<std::string, double>* name_to_priority, std::string* schedule_node_name);
    static void GetUpdateSequence(std::string& input_name, std::map<std::string, CostNode> node_map,
                                  std::map<std::string, int64_t> name_to_indegree,
                                  std::vector<std::string>* update_sequence);
    static void InitStartAndEndTime(std::vector<std::string> update_sequence, std::map<std::string, CostNode> node_map,
                                    std::map<std::string, int64_t>* earlist_start_times,
                                    std::map<std::string, int64_t>* latest_start_times, int64_t* total_time);
    static cpp::result<void, Error> GetBestDevice(CostNode schedule_node, std::map<std::string, CostNode> node_map,
                                                  std::map<std::string, Device>* device_map, std::string* best_device,
                                                  int64_t* execute_time);
    static void ScheduleNodeToDevice(CostNode schedule_node, std::string& best_device, int64_t execute_time,
                                     std::map<std::string, CostNode>* node_map,
                                     std::map<std::string, Device>* device_map,
                                     std::map<std::string, std::string>* node_to_device);
    static void UpdateStartAndEndTime(std::vector<std::string> update_sequence,
                                      std::map<std::string, int64_t> scheduled_nodes, int64_t total_time,
                                      std::map<std::string, CostNode>* node_map,
                                      std::map<std::string, int64_t>* earlist_start_times,
                                      std::map<std::string, int64_t>* latest_start_times);
    static void UpdateScheduleableNodes(CostNode schedule_node, std::map<std::string, int64_t>* name_to_indegree,
                                        std::map<std::string, double>* name_to_priority);

    DECL_ACCESSOR(GetCostGraph, SetCostGraph, cost_graph, M)
    DECL_ACCESSOR(GetDevices, SetDevices, devices, M)
};
}  // namespace geesibling

#endif
