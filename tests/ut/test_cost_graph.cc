#include <string>

#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "common/util.hpp"
#include "cost_graph/common.hpp"
#include "cost_graph/cost_graph.hpp"
#include "gtest/gtest.h"

namespace geesibling {
TEST(TestCostGraph, CostGraph) {
    NodeBase node_1;
    node_1.Name("input");
    node_1.Device("dev0");
    node_1.ComputeCost(2);
    node_1.PersistentMemory(1);
    node_1.Outputs({"conv1", "conv2"});
    node_1.OutputMemory(10);

    NodeBase node_2;
    node_2.Name("conv1");
    node_2.Device("dev1");
    node_2.ComputeCost(3);
    node_2.Inputs({"input"});
    node_2.InputMemory(2);
    node_2.PersistentMemory(3);
    node_2.Outputs({"conv2"});
    node_2.OutputMemory(3);

    NodeBase node_3;
    node_3.Name("conv2");
    node_3.Device("dev2");
    node_3.ComputeCost(3);
    node_3.Inputs({"input", "conv1"});
    node_3.InputMemory(8);
    node_3.PersistentMemory(4);
    node_3.Outputs({"conv3"});
    node_3.OutputMemory(3);

    NodeBase node_4;
    node_4.Name("conv3");
    node_4.Device("dev0");
    node_4.ComputeCost(3);
    node_4.Inputs({"conv2"});
    node_4.InputMemory(3);
    node_4.PersistentMemory(4);
    node_4.Outputs({"relu"});
    node_4.OutputMemory(3);

    NodeBase node_5;
    node_5.Name("relu");
    node_5.Device("dev1");
    node_5.ComputeCost(3);
    node_5.Inputs({"conv3"});
    node_5.InputMemory(3);
    node_5.PersistentMemory(4);
    node_5.Outputs({"output"});
    node_5.OutputMemory(3);

    NodeBase node_6;
    node_6.Name("output");
    node_6.Device("dev3");
    node_6.ComputeCost(3);
    node_6.Inputs({"relu"});
    node_6.InputMemory(3);
    node_6.PersistentMemory(6);

    Graph graph;
    graph.AddNode(node_1);
    graph.AddNode(node_2);
    graph.AddNode(node_3);
    graph.AddNode(node_4);
    graph.AddNode(node_5);
    graph.AddNode(node_6);

    EXPECT_EQ(6, graph.Nodes().size());
    EXPECT_EQ(6, graph.NodeMap().size());

    CostGraph cost_graph = ConvertGraphToCostGraph(graph);

    EXPECT_EQ(6, cost_graph.GetCostNodes().size());
    EXPECT_EQ(6, cost_graph.GetCostNodeMap().size());
    EXPECT_EQ(5, cost_graph.GetCostNodes()[0].GetOutputCommCosts()[0]);
    EXPECT_EQ(4, cost_graph.GetCostNodes()[2].GetInputCommCosts()[0]);

    MergedCostGraph merged_cost_graph = InitMergedCostGraph(cost_graph);

    EXPECT_EQ(6, merged_cost_graph.GetMergedCostNodes().size());
    EXPECT_EQ(6, merged_cost_graph.GetMergedCostNodeMap().size());
    EXPECT_EQ("input", merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[0]);
    EXPECT_EQ("conv1", merged_cost_graph.GetMergedCostNodes()[1].GetCostNodeNames()[0]);
    EXPECT_EQ("output", merged_cost_graph.GetMergedCostNodes()[5].GetCostNodeNames()[0]);

    CostGraph new_cost_graph = ConvertMergedCostGraphToCostGraph(merged_cost_graph);

    EXPECT_EQ(6, new_cost_graph.GetCostNodes().size());
    EXPECT_EQ(6, new_cost_graph.GetCostNodeMap().size());
    EXPECT_EQ(5, new_cost_graph.GetCostNodes()[0].GetOutputCommCosts()[0]);
    EXPECT_EQ(4, new_cost_graph.GetCostNodes()[2].GetInputCommCosts()[0]);
}
}  // namespace geesibling