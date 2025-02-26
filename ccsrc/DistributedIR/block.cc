#include "DistributedIR/block.hpp"

namespace geesibling {

void DeviceGraph::Connect(int start_item, int start_out_index, int end_item, int end_arg_index) {
    edges.emplace_back(blocks[start_item], start_out_index, blocks[end_item], end_arg_index);
}

void ServerGraph::Connect(int start_item, int start_out_index, int end_item, int end_arg_index) {
    edges.emplace_back(device_graphs[start_item], start_out_index, device_graphs[end_item], end_arg_index);
}

void ClusterGraph::Connect(int start_item, int start_out_index, int end_item, int end_arg_index) {
    edges.emplace_back(server_graphs[start_item], start_out_index, server_graphs[end_item], end_arg_index);
}

}  // namespace geesibling
