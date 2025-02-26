#include <algorithm>
#include <cstdint>

#include "Builder.h"
#include "cluster/server.hpp"

#ifndef GEESIBLING_SGD_ALGORITHM_H
#define GEESIBLING_SGD_ALGORITHM_H

namespace geesibling {

class Partition {
  public:
    Builder builder = Builder();
    std::vector<GraphGroup> group_list;
    std::map<std::string, std::vector<std::string> > group_info;
    std::int64_t group_balance_cost;
    std::vector<SgpEdge> critical_edges;
    std::map<std::string, std::int64_t> op_group_dict;
    std::map<std::string, std::string> op_group;

    static std::int64_t GetOpMemory(Graph& graph, const std::string& op);
    Partition(Graph& graph, std::int64_t group_num, std::vector<Device> devices, float target_factor, int OOM);
    void GraphPlacement(Graph& graph, std::vector<Device> devices, std::int64_t max_iterations = 500,
                        float target_factor = 0.6);
    void Split(Graph& graph);
    static bool Cmp(const SgpEdge& e1, const SgpEdge& e2);
    void AdjustV2(Graph& graph, std::int64_t max_iterations, float target_factor,
                  std::map<int64_t, std::string> id_device_map);
};
}  // namespace geesibling

#endif
