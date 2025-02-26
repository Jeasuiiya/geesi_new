#pragma once

#include <algorithm>
#ifndef GEESIBLING_IR_EDGE_H
#define GEESIBLING_IR_EDGE_H
#include <functional>
#include <iostream>
#include <string>

#include "fmt/format.h"
namespace geesibling {
template <typename T>
struct Edge;
template <typename T>
struct EdgePort {
    static_assert(!std::is_reference_v<T> && !std::is_pointer_v<T>, "T must not be a reference or pointer type.");
    friend struct fmt::formatter<EdgePort<T>>;
    T entity;
    // port index
    int index;
    Edge<T>* edge = nullptr;
    EdgePort(const T& entity, int index) : entity(entity), index(index) {}

    EdgePort(T&& entity, int index) : entity(std::move(entity)), index(index) {}
    bool operator==(const EdgePort<T> port) const {
        return entity == port.entity && this->index == port.index;
    }

    EdgePort<T> operator|(const Edge<T>& edge) {
        this->edge = &edge;
        return *this;
    }

    EdgePort<T> operator|(Edge<T>* edge) {
        this->edge = edge;
        return *this;
    }

    EdgePort<T> operator>>(Edge<T>& edge) {
        edge.start = *this;
        return *this;
    }

    EdgePort<T> operator<<(Edge<T>& edge) {
        edge.end = *this;
        return *this;
    }

    EdgePort<T> operator>>(Edge<T>* edge) {
        edge->start = *this;
        return *this;
    }

    EdgePort<T> operator<<(Edge<T>* edge) {
        edge->end = *this;
        return *this;
    }
};

template <>
struct EdgePort<std::string> {
    friend struct fmt::formatter<EdgePort<std::string>>;
    std::string entity;
    explicit EdgePort(std::string entity) : entity(std::move(entity)) {}
    bool operator==(const EdgePort<std::string>& port) const {
        return this->entity == port.entity;
    }
};

template <typename T>
struct Edge {
    friend struct fmt::formatter<Edge<T>>;
    static_assert(!std::is_reference_v<T> && !std::is_pointer_v<T>, "T must not be a reference or pointer type.");

    Edge(T start, int start_index, T end, int end_index)
        : start(EdgePort<T>(start, start_index)), end(EdgePort<T>(end, end_index)) {
        this->start | this;
        this->end | this;
    }
    Edge(EdgePort<T> start, EdgePort<T> end) : start(start), end(end) {
        this->start | this;
        this->end | this;
    }
    EdgePort<T> start;
    EdgePort<T> end;
};

struct HasInternalEdge {
    virtual void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) = 0;
};

template <typename T>
struct HasEdgePort {
    virtual void BuildInputPorts() = 0;
    virtual void BuildOutputPorts() = 0;
    void BuildSeclectedPorts(const std::function<uint64_t(const T&)>& searchPortSize,
                             const std::function<EdgePort<T>(Edge<T>&)>& getEdgePort, std::vector<T>& internelEles,
                             std::vector<Edge<T>>& edges, std::vector<EdgePort<T>>& out) {
        std::vector<EdgePort<T>> result;
        for (auto& ele : internelEles) {
            for (size_t i = 0; i < searchPortSize(ele); i++) {
                EdgePort<T> ep(ele, i);
                auto find = std::find_if(edges.begin(), edges.end(), [&](auto& e) { return getEdgePort(e) == ep; });
                if (find == edges.end()) {
                    result.push_back(ep);
                }
            }
        }
        out = result;
    }
    virtual void BuildPorts() {
        BuildInputPorts();
        BuildOutputPorts();
    }
};
}  // namespace geesibling

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
struct fmt::formatter<geesibling::EdgePort<T>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const geesibling::EdgePort<T>& ep, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "<{:s}, {}>", ep.entity, ep.index);
    }
};

template <>
struct fmt::formatter<geesibling::EdgePort<std::string>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const geesibling::EdgePort<std::string>& ep, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "<{}>", ep.entity);
    }
};

template <typename T>
struct fmt::formatter<geesibling::Edge<T>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const geesibling::Edge<T>& e, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Edge(start={}, end={})", e.start, e.end);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif
