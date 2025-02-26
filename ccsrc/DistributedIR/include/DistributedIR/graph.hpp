#ifndef GEESIBLING_GRAPH_GRAPH_H
#define GEESIBLING_GRAPH_GRAPH_H

#include <memory>
#include <string>
#include <vector>

#include "common/fmt.hpp"
#include "common/types.hpp"
#include "common/util.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "node.hpp"
namespace geesibling {
class SubGraph;
using SubGraphPtr = std::shared_ptr<SubGraph>;
using SubGraphWeakPtr = std::weak_ptr<SubGraph>;
class Graph {
    friend struct fmt::formatter<Graph>;
    friend struct fmt::formatter<SubGraph>;
    friend struct fmt::formatter<SubGraphPtr>;

  private:
    std::vector<NodePtr> nodes;
    std::map<std::string, NodePtr> node_map;
    std::vector<StrAndInt> returns;

  public:
    Graph() = default;
    virtual ~Graph() = default;
    Graph(const Graph& g) {
        for (const auto& i : g.nodes) {
            auto ptr = std::make_shared<NodeBase>(*i);
            nodes.push_back(ptr);
            node_map.insert({ptr->Name(), ptr});
        }
    }
    Graph(Graph&& g) noexcept : nodes(std::move(g.nodes)), node_map(std::move(g.node_map)) {}
    DECL_ACCESSOR(Nodes, Nodes, nodes, M)
    DECL_ACCESSOR(NodeMap, NodeMap, node_map, M)
    // DECL_ACCESSOR(Outputs,  Outputs, outputs, M)
    void AddNode(NodeBase node) {
        auto n = std::make_shared<NodeBase>(node);
        node_map.insert({node.Name(), n});
        nodes.push_back(n);
    }
    void AddNode(int at, NodeBase node) {
        auto n = std::make_shared<NodeBase>(node);
        node_map.insert({node.Name(), n});
        nodes.insert(nodes.begin() + at, n);
    }

    void AddNode(const NodePtr& node) {
        node_map.insert({node->Name(), node});
        nodes.push_back(node);
    }
    void AddNode(int at, const NodePtr& node) {
        node_map.insert({node->Name(), node});
        nodes.insert(nodes.begin() + at, node);
    }

    cpp::result<NodePtr&, Error> GetNode(uint at) {
        if (at >= nodes.size()) {
            return cpp::fail<Error>(Kind::Invalid, "out of range");
        }
        return nodes.at(at);
    }
    cpp::result<NodePtr&, Error> GetNode(const std::string& name) {
        auto find = node_map.find(name);
        if (find == node_map.end()) {
            return cpp::fail<Error>(Kind::Invalid, fmt::format("no such node: {}", name));
        }
        return node_map.find(name)->second;
    }
    int GetNodesNum() {
        return nodes.size();
    }
    void AddReturn(const StrAndInt& r) {
        returns.push_back(r);
    }
    void ClearReturn() {
        returns.clear();
    }
    DECL_GETTER(Returns, returns)
    void ClearNode() {
        node_map.clear();
        nodes.clear();
    }
};
class SubGraph : public Graph {
    friend struct fmt::formatter<SubGraph>;
    friend struct fmt::formatter<SubGraphPtr>;

    std::vector<SubGraphPtr> input_graphs;                              // 输入图
    std::vector<std::vector<std::pair<StrAndInt, StrAndInt>>> inputs;   // 各图输入
    std::vector<SubGraphPtr> output_graphs;                             // 输出图
    std::vector<std::vector<std::pair<StrAndInt, StrAndInt>>> outputs;  // 输出
    // std::vector<std::multimap<StrAndInt, StrAndInt>> outputs;          // 输出
  public:
    SubGraph() = default;
    SubGraph(const SubGraph& g) = default;
    SubGraph& operator=(const SubGraph& g) {
        input_graphs = g.input_graphs;
        inputs = g.inputs;
        output_graphs = g.output_graphs;
        outputs = g.outputs;
        return *this;
    }
    std::string Device() {
        // SubGraph has 1 node at least.
        auto b = !Nodes().empty();
        assert(b);
        auto r = GetNode(0);
        assert(r.has_value());
        return r.value()->Device();
    }
    void AddInputGraph(const SubGraphPtr& g) {
        input_graphs.push_back(g);
    }
    void AddInputGraph(const SubGraph& g) {
        AddInputGraph(std::make_shared<SubGraph>(g));
    }
    void AddOutputGraph(const SubGraphPtr& g) {
        output_graphs.push_back(g);
    }
    void AddOutputGraph(const SubGraph& g) {
        AddOutputGraph(std::make_shared<SubGraph>(g));
    }

    void AddInput(const std::vector<std::pair<StrAndInt, StrAndInt>>& op_op) {
        inputs.push_back(op_op);
    }
    void AddOutput(const std::vector<std::pair<StrAndInt, StrAndInt>>& op_op) {
        outputs.push_back(op_op);
    }

    // using Graph::AddReturn;
    // using Graph::ClearReturn;
    // using Graph::Returns;
    DECL_GETTER(GetInputs, inputs)
    DECL_GETTER(GetInputGraphs, input_graphs)
    DECL_GETTER(GetOutputs, outputs)
    DECL_GETTER(GetOutputGraphs, output_graphs)
};

}  // namespace geesibling

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<geesibling::Graph> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const geesibling::Graph& g, FormatContext& ctx) const -> decltype(ctx.out()) {
        auto nodes = g.nodes | ranges::views::transform([](const auto& i) { return fmt_shared(i); });
        return fmt::format_to(ctx.out(), "Graph(nodes={}, outputs={})", nodes, g.returns);
    }
};

template <>
struct fmt::formatter<geesibling::SubGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::SubGraph& g, FormatContext& ctx) const -> decltype(ctx.out()) {
        auto nodes = g.nodes | ranges::views::transform([](const auto& i) { return fmt_shared(i); });
        {
            if (presentation == 's') {
                auto input_graphs =
                    g.input_graphs | ranges::views::transform([](auto& i) { return fmt::ptr(i.get()); });
                auto output_graphs =
                    g.output_graphs | ranges::views::transform([](auto& i) { return fmt::ptr(i.get()); });
                return fmt::format_to(
                    ctx.out(), "SubGraph({}, nodes={}, input_graphs={}, inputs={}, output_graphs={}, outputs={})",
                    fmt::ptr(&g), nodes, input_graphs, g.inputs, output_graphs, g.outputs);
            }
        }
        auto input_graphs = g.input_graphs | ranges::views::transform([](const auto& i) { return fmt_shared(i); });
        auto output_graphs = g.output_graphs | ranges::views::transform([](const auto& i) { return fmt_shared(i); });
        return fmt::format_to(ctx.out(),
                              "SubGraph({}, nodes={}, input_graphs={}, inputs={}, output_graphs={}, outputs={})",
                              fmt::ptr(&g), nodes, input_graphs, g.inputs, output_graphs, g.outputs);
    }
};
// NOLINTEND(readability-identifier-naming)

#endif
