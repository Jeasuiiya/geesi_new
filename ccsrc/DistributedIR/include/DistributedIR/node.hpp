#ifndef GEESIBLING_GRAPH_NODE_H
#define GEESIBLING_GRAPH_NODE_H

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DistributedIR/edge.hpp"
#include "DistributedIR/input.hpp"
#include "DistributedIR/tensor.hpp"
#include "common/error.hpp"
#include "common/result_macro.hpp"
#include "common/types.hpp"
#include "common/util.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "range/v3/algorithm/find_if.hpp"
#include "range/v3/all.hpp"
#include "result.hpp"

namespace geesibling {

class NodeBase;
using NodePtr = std::shared_ptr<NodeBase>;

class NodeBase {
    friend struct fmt::formatter<NodeBase>;
    friend struct fmt::formatter<NodePtr>;

  private:
    std::string name;  // node name
    std::string op;    // op name
    // intpus and outputs just record node name which this node is connected with.
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<EdgePort<InputStr>> input_ports;
    std::vector<EdgePort<AbstractTensor>> output_ports;
    std::string device;                        // 该节点的计算设备
    std::map<std::string, std::string> attrs;  // 节点属性
    int64_t outputs_num{0};                    // 输出个数
    int64_t start_time{0};                     // 开始时间
    int64_t end_time{0};                       // 结束时间
    int64_t compute_cost{0};                   // 计算代价
    int64_t temporary_memory{0};               // 临时内存
    int64_t persistent_memory{0};              // 持久内存
    int64_t input_memory{0};                   // 输入内存
    int64_t output_memory{0};                  // 输出内存

    // T data;
  public:
    NodeBase() = default;

    DECL_ACCESSOR(Name, Name, name, M)
    DECL_ACCESSOR(Op, Op, op, M)
    DECL_ACCESSOR(Device, Device, device, C)
    DECL_ACCESSOR(Inputs, Inputs, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, outputs, M)
    DECL_ACCESSOR(OutputsNum, OutputsNum, outputs_num, M)
    DECL_ACCESSOR(Attrs, Attrs, attrs, M)
    DECL_ACCESSOR(StartTime, StartTime, start_time, M)
    DECL_ACCESSOR(EndTime, EndTime, end_time, M)
    DECL_ACCESSOR(ComputeCost, ComputeCost, compute_cost, M)
    DECL_ACCESSOR(TemporaryMemory, TemporaryMemory, temporary_memory, M)
    DECL_ACCESSOR(PersistentMemory, PersistentMemory, persistent_memory, M)
    DECL_ACCESSOR(InputMemory, InputMemory, input_memory, M)
    DECL_ACCESSOR(OutputMemory, OutputMemory, output_memory, M)
    DECL_GETTER(InputPorts, input_ports)
    DECL_GETTER(OutputPorts, output_ports)

    void AddInput(const std::string& input) {
        inputs.push_back(input);
    }
    void AddOutput(const std::string& output) {
        outputs.push_back(output);
    }

    cpp::result<EdgePort<InputStr>&, Error> InputPort(int index) {
        auto r = ranges::find_if(input_ports, [=](const EdgePort<InputStr>& i) { return i.index == index; });
        if (r == input_ports.end()) {
            return cpp::fail(Error(Kind::Invalid, fmt::format("Input EdgePort for index {} is not exist.", index)));
        }
        return *r;
    }
    cpp::result<StrAndInt, Error> InputRef(int index) {
        TRY_ASSIGN(auto r, InputPort(index));
        return r.entity.Ref();
    }

    cpp::result<void, Error> AddInputPort(const std::string& input_node, int input_index, int index, DataType dtype,
                                          const shape_t& shape) {
        EdgePort<InputStr> p(InputStr(input_node, input_index, AbstractTensor(dtype, shape)), index);
        auto r = InputPort(p.index);
        if (r.has_value()) {
            return cpp::fail(Error(Kind::Invalid, fmt::format("Input EdgePort for index {} is exist.", p.index)));
        }
        input_ports.emplace_back(p);
        SortInputPort();
        return {};
    }
    template <class T = EdgePort<InputStr>, class = std::enable_if_t<std::is_same_v<T, EdgePort<InputStr>>>>
    cpp::result<void, Error> AddInputPort(T&& p) {
        auto r = InputPort(p.index);
        if (r.has_value()) {
            return cpp::fail(Error(Kind::Invalid, fmt::format("Input EdgePort for index {} is exist.", p.index)));
        }
        input_ports.emplace_back(std::forward<T>(p));
        SortInputPort();
        return {};
    }

    void DelInputPort(int index) {
        input_ports = input_ports
                      | ranges::views::remove_if([=](const EdgePort<InputStr>& p) { return p.index == index; })
                      | ranges::to<decltype(input_ports)>();
    }

    void SortInputPort() {
        ranges::sort(input_ports, [](const auto& a1, const auto& a2) { return a1.index < a2.index; });
    }
    cpp::result<void, Error> AddOutputPort(DataType dtype, const shape_t& shape, int index) {
        return AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(dtype, shape), index));
    }
    cpp::result<void, Error> AddOutputPort(DataType dtype, const shape_t& shape, int index, void* ptr, size_t length) {
        return AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(dtype, shape, ptr, length), index));
    }
    template <class T = EdgePort<AbstractTensor>, class = std::enable_if_t<std::is_same_v<T, EdgePort<AbstractTensor>>>>
    cpp::result<void, Error> AddOutputPort(T&& p) {
        auto r = OutputPort(p.index);
        if (r.has_value()) {
            return cpp::fail(Error(Kind::Invalid, fmt::format("Output EdgePort for index {} is exist.", p.index)));
        }
        output_ports.emplace_back(std::forward<T>(p));
        SortOutputPort();
        return {};
    }

    void DelOutputPort(int index) {
        output_ports = output_ports
                       | ranges::views::remove_if([=](const EdgePort<AbstractTensor>& p) { return p.index == index; })
                       | ranges::to<decltype(output_ports)>();
    }

    void SortOutputPort() {
        ranges::sort(output_ports, [](const auto& a1, const auto& a2) { return a1.index < a2.index; });
    }
    size_t InputSize() {
        return input_ports.size();
    }
    size_t OutputSize() {
        return output_ports.size();
    }

    cpp::result<StrAndInt, Error> InputName(int index) {
        TRY(InputPort(index));
        return std::make_pair(name, index);
    }

    cpp::result<StrAndInt, Error> OutputName(int index) {
        TRY(OutputPort(index));
        return std::make_pair(name, index);
    }

    cpp::result<EdgePort<AbstractTensor>&, Error> OutputPort(int index) {
        auto r = ranges::find_if(output_ports, [=](const EdgePort<AbstractTensor>& i) { return i.index == index; });
        if (r == output_ports.end()) {
            return cpp::fail(Error(Kind::Invalid, fmt::format("Output EdgePort for index {} is not exist.", index)));
        }
        return *r;
    }
};

class MergedNode : public NodeBase {
    std::vector<NodeBase> merged_nodes;  // 已合并节点
};

}  // namespace geesibling

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<geesibling::NodeBase> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const geesibling::NodeBase& n, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(
            ctx.out(),
            "NodeBase(name={}, op={}, device={}, inputs={}, outputs={}, input_ports={}, output_ports={}, attrs={})",
            n.name, n.op, n.device, n.inputs, n.outputs, n.input_ports, n.output_ports, n.attrs);
    }
};
// NOLINTEND(readability-identifier-naming)

#endif /* ifndef _GRAPH_NODE_H */
