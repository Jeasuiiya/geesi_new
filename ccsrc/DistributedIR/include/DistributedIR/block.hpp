#pragma once

#ifndef GEESIBLING_IR_BLOCK_H
#define GEESIBLING_IR_BLOCK_H
#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DistributedIR/graph.hpp"
#include "common/fmt.hpp"
#include "common/id.hpp"
#include "common/util.hpp"
#include "edge.hpp"
namespace geesibling {

class Block {
    friend struct fmt::formatter<Block>;
    // if source is 0, it represents source is a external input, such user input args
    using BlockInput = Input<int64_t>;
    using BlockOutput = AbstractTensor;
    using BlockInputPort = EdgePort<BlockInput>;
    using BlockOutputPort = EdgePort<BlockOutput>;

  private:
    explicit Block() {
        id = IDGenerator.Gen();
    }
    int64_t id;
    std::string device;
    SubGraphPtr graph;
    std::vector<BlockInputPort> inputs;
    std::vector<BlockOutputPort> outputs;

  public:
    explicit Block(const std::string& device) : Block() {
        this->device = device;
    }
    explicit Block(SubGraph& graph) : Block(graph.Device()) {
        this->graph = std::make_shared<SubGraph>(graph);
    }
    explicit Block(SubGraphPtr& graph) : Block(graph->Device()) {
        this->graph = graph;
    }

    virtual ~Block() = default;

    void AddInputPort(const BlockInputPort& port) {
        inputs.push_back(port);
    }
    void AddInputPort(const BlockInput& input) {
        // default new input is the latest one
        AddInputPort(BlockInputPort(input, inputs.size()));
    }
    void AddInputPort(int64_t blockId, int index, DataType dtype, const shape_t& shape) {
        AddInputPort(BlockInput(blockId, index, AbstractTensor(dtype, shape)));
    }
    void AddOutputPort(const BlockOutputPort& port) {
        outputs.push_back(port);
    }
    void AddOutputPort(const BlockOutput& output) {
        // default new output is the latest one
        AddOutputPort(BlockOutputPort(output, outputs.size()));
    }
    void AddOutputPort(DataType dtype, const shape_t& shape) {
        // default new output is the latest one
        AddOutputPort(AbstractTensor(dtype, shape));
    }
    DECL_ACCESSOR(Inputs, Inputs, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, outputs, M)
    DECL_GETTER(Id, id)
    DECL_GETTER(Device, device)
    DECL_GETTER(GetSubGraph, graph)
    bool operator==(const Block& block) const {
        return id == block.id;
    }
};

class DeviceGraph : public HasInternalEdge, public HasEdgePort<Block> {
    friend struct fmt::formatter<DeviceGraph>;

  private:
    std::string id;
    std::string device;
    std::vector<Block> blocks;
    // internal edge
    std::vector<Edge<Block>> edges;

    // port
    std::vector<EdgePort<Block>> inputs;
    std::vector<EdgePort<Block>> outputs;

  public:
    DeviceGraph() = default;
    explicit DeviceGraph(std::string id) : id(std::move(id)){};
    virtual ~DeviceGraph() = default;
    DECL_ACCESSOR(Inputs, Inputs, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, outputs, M)
    DECL_GETTER(GetBlocks, blocks)
    DECL_GETTER(GetEdges, edges)
    void AddBlock(const Block& block) {
        blocks.emplace_back(block);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;

    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search inputs
                return b.Inputs().size();
            },
            [](auto e) {
                // input edge port must be a edge end
                return e.end;
            },
            blocks, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                // output edge port must be a edge start
                return e.start;
            },
            blocks, edges, outputs);
    }
    bool operator==(const DeviceGraph& graph) const {
        return id == graph.id;
    }
};

class ServerGraph : public HasInternalEdge, public HasEdgePort<DeviceGraph> {
    friend struct fmt::formatter<ServerGraph>;

  private:
    std::string id;
    std::string server;
    std::vector<DeviceGraph> device_graphs;
    std::vector<Edge<DeviceGraph>> edges;
    // port
    std::vector<EdgePort<DeviceGraph>> inputs;
    std::vector<EdgePort<DeviceGraph>> outputs;

  public:
    ServerGraph() = default;
    explicit ServerGraph(std::string id) : id(std::move(id)){};
    virtual ~ServerGraph() = default;
    DECL_ACCESSOR(Inputs, Inputs, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, outputs, M)
    DECL_GETTER(GetDeviceGraphs, device_graphs)
    DECL_GETTER(GetEdges, edges)
    void AddDeviceGraph(const DeviceGraph& graph) {
        device_graphs.emplace_back(graph);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;
    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Inputs().size();
            },
            [](auto e) {
                // input edge port must be a edge end
                return e.end;
            },
            device_graphs, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                return
                    // output edge port must be a edge start
                    e.start;
            },
            device_graphs, edges, outputs);
    }

    bool operator==(const ServerGraph& graph) const {
        return id == graph.id;
    }
    std::string ToString() {
        std::stringstream ss;
        ss << "server_graph_id:" << id << std::endl;
        return ss.str();
    }
};

class ClusterGraph : public HasInternalEdge, public HasEdgePort<ServerGraph> {
    friend struct fmt::formatter<ClusterGraph>;

  private:
    std::string id;
    std::vector<ServerGraph> server_graphs;
    std::vector<Edge<ServerGraph>> edges;
    // port
    std::vector<EdgePort<ServerGraph>> inputs;
    std::vector<EdgePort<ServerGraph>> outputs;

  public:
    ClusterGraph() = default;
    explicit ClusterGraph(std::string id) : id(std::move(id)){};
    virtual ~ClusterGraph() = default;
    DECL_GETTER(GetServerGraphs, server_graphs)
    DECL_GETTER(GetEdges, edges)
    void AddServerGraph(const ServerGraph& graph) {
        server_graphs.emplace_back(graph);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;
    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Inputs().size();
            },
            [](auto e) {
                return
                    // input edge port must be a edge end
                    e.end;
            },
            server_graphs, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                // output edge port must be a edge start
                return e.start;
            },
            server_graphs, edges, outputs);
    }

    bool operator==(const ClusterGraph& graph) const {
        return id == graph.id;
    }
    std::string ToString() {
        std::stringstream ss;
        ss << "cluster_graph_id:" << id << std::endl;
        return ss.str();
    }
};

}  // namespace geesibling
// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<geesibling::Block> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::Block& b, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "Block(id={}, device={})", b.id, b.device);
        }
        return fmt::format_to(ctx.out(), "Block(id={}, device={}, graph={:s}, inputs={}, outputs={})", b.id, b.device,
                              b.graph, b.inputs, b.outputs);
    }
};

template <>
struct fmt::formatter<geesibling::DeviceGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::DeviceGraph& dg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "DeviceGraph(id={})", dg.id);
        }
        return fmt::format_to(ctx.out(), "DeviceGraph(id={}, device={}, blocks={}, edges={}, inputs={}, outputs={})",
                              dg.id, dg.device, dg.blocks, dg.edges, dg.inputs, dg.outputs);
    }
};

template <>
struct fmt::formatter<geesibling::ServerGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::ServerGraph& sg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "ServerGraph(id={})", sg.id);
        }
        return fmt::format_to(ctx.out(), "ServerGraph(id={}, device={}, graphs={}, edges={}, inputs={}, outputs={})",
                              sg.id, sg.server, sg.device_graphs, sg.edges, sg.inputs, sg.outputs);
    }
};
template <>
struct fmt::formatter<geesibling::ClusterGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::ClusterGraph& cg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "ClusterGraph(id={})", cg.id);
        }
        return fmt::format_to(ctx.out(), "ClusterGraph(id={}, graphs={}, edges={}, inputs={}, outputs={})", cg.id,
                              cg.server_graphs, cg.edges, cg.inputs, cg.outputs);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif /* end of include guard: GEESIBLING_IR_BLOCK_H */
