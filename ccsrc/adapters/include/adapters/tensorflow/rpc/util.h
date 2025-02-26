#include <map>
#include <string>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/graph.pb.h"
#ifndef ADAPTERS_TENSORFLOW_RPC_RPC_UTIL_H
#define ADAPTERS_TENSORFLOW_RPC_RPC_UTIL_H

namespace geesibling {

std::map<std::string, std::string> GetDeviceMapFromMessage(geesibling::rpc::Graph const& graph);

geesibling::rpc::Graph ConvertGraphToMessage(geesibling::Graph& graph);

geesibling::Graph ConvertMessageToGraph(const geesibling::rpc::Graph& graph);
}  // namespace geesibling
#endif
