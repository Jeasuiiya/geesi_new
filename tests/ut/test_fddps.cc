#include "cost_graph/common.hpp"
#include "gtest/gtest.h"
#include "policy/fd-dps/fddps_algorithm.h"

namespace geesibling {

TEST(TestFDDPS, FDDPS) {
    NodeBase node_1;
    node_1.Name("input");
    node_1.Device("/GPU:0");
    node_1.ComputeCost(2);
    node_1.PersistentMemory(1);
    node_1.Outputs({"conv1", "conv2", "gelu"});
    node_1.OutputMemory(0);
    node_1.StartTime(0);
    node_1.EndTime(2);

    NodeBase node_2;
    node_2.Name("conv1");
    node_2.Device("/GPU:1");
    node_2.ComputeCost(5);
    node_2.Inputs({"input"});
    node_2.InputMemory(0);
    node_2.PersistentMemory(3);
    node_2.Outputs({"conv2"});
    node_2.OutputMemory(0);
    node_2.StartTime(2);
    node_2.EndTime(7);

    NodeBase node_3;
    node_3.Name("gelu");
    node_3.Device("/GPU:3");
    node_3.ComputeCost(10);
    node_3.Inputs({"input"});
    node_3.InputMemory(0);
    node_3.PersistentMemory(4);
    node_3.Outputs({"conv2"});
    node_3.OutputMemory(0);
    node_3.StartTime(2);
    node_3.EndTime(12);

    NodeBase node_4;
    node_4.Name("conv2");
    node_4.Device("/GPU:2");
    node_4.ComputeCost(3);
    node_4.Inputs({"input", "conv1", "gelu"});
    node_4.InputMemory(0);
    node_4.PersistentMemory(4);
    node_4.Outputs({"conv3"});
    node_4.OutputMemory(0);
    node_4.StartTime(12);
    node_4.EndTime(15);

    NodeBase node_5;
    node_5.Name("conv3");
    node_5.Device("/GPU:1");
    node_5.ComputeCost(3);
    node_5.Inputs({"conv2"});
    node_5.InputMemory(0);
    node_5.PersistentMemory(4);
    node_5.Outputs({"relu"});
    node_5.OutputMemory(0);
    node_5.StartTime(15);
    node_5.EndTime(18);

    NodeBase node_6;
    node_6.Name("relu");
    node_6.Device("/GPU:1");
    node_6.ComputeCost(3);
    node_6.Inputs({"conv3"});
    node_6.InputMemory(0);
    node_6.PersistentMemory(4);
    node_6.Outputs({"output"});
    node_6.OutputMemory(0);
    node_6.StartTime(18);
    node_6.EndTime(21);

    NodeBase node_7;
    node_7.Name("output");
    node_7.Device("/GPU:3");
    node_7.ComputeCost(3);
    node_7.Inputs({"relu"});
    node_7.InputMemory(0);
    node_7.PersistentMemory(6);
    node_7.StartTime(21);
    node_7.EndTime(24);

    Graph graph;
    graph.AddNode(node_1);
    graph.AddNode(node_2);
    graph.AddNode(node_3);
    graph.AddNode(node_4);
    graph.AddNode(node_5);
    graph.AddNode(node_6);
    graph.AddNode(node_7);

    Device device_1(DeviceType::NVGpu, "/GPU:0", 20, 20, 0);
    Device device_2(DeviceType::NVGpu, "/GPU:1", 20, 20, 0);
    Device device_3(DeviceType::NVGpu, "/GPU:2", 20, 20, 0);
    Device device_4(DeviceType::NVGpu, "/GPU:3", 20, 20, 0);
    std::vector<Device> devices = {device_1, device_2, device_3, device_4};

    CostGraph cost_graph = ConvertGraphToCostGraph(graph);
    FDDPSAlgorithm fddps_algorithm(cost_graph, devices);
    auto r = fddps_algorithm.Placement();
}
}  // namespace geesibling
