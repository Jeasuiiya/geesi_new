#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <DistributedIR/block.hpp>
#include <DistributedIR/graph.hpp>
#include <memory>
#include <utility>

#include "DistributedIR/dividegraph.hpp"
#include "cluster/server.hpp"
#include "common/fmt.hpp"
#include "common/log.h"
#include "common/types.hpp"
#include "cost_graph/common.hpp"
#include "cost_graph/cost_graph.hpp"
#include "fmt/format.h"
#include "policy/fd-dps/fddps_algorithm.h"
#include "policy/sgp/graphPartition.h"
#include "range/v3/algorithm/transform.hpp"
#include "range/v3/range/conversion.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace geesibling::py {

class Node {
  private:
    std::shared_ptr<geesibling::NodeBase> node_ptr;

  public:
    explicit Node(const geesibling::NodeBase& node) {
        node_ptr = std::make_shared<geesibling::NodeBase>(node);
    }
    explicit Node(std::shared_ptr<geesibling::NodeBase> node) {
        node_ptr = std::move(node);
    }
    Node(const Node& node) {
        node_ptr = node.node_ptr;
    }
    Node() {
        node_ptr = std::make_shared<geesibling::NodeBase>();
    }
    ~Node() = default;
    std::shared_ptr<geesibling::NodeBase>& NodePtr() {
        return this->node_ptr;
    }
    DECL_ACCESSOR_PROXY_S(SetName, GetName, std::string, node_ptr, Name, M)
    DECL_ACCESSOR_PROXY_S(SetOp, GetOp, std::string, node_ptr, Op, M)
    DECL_ACCESSOR_PROXY_S(SetDevice, GetDevice, std::string, node_ptr, Device, M)
    DECL_ACCESSOR_PROXY_S(SetInputs, GetInputs, std::vector<std::string>, node_ptr, Inputs, M)
    DECL_ACCESSOR_PROXY_S(SetOutputs, GetOutputs, std::vector<std::string>, node_ptr, Outputs, M)
    DECL_ACCESSOR_PROXY_S(SetAttrs, GetAttrs, ALL(std::map<std::string, std::string>), node_ptr, Attrs, M)
    DECL_ACCESSOR_PROXY_S(SetStartTime, GetStartTime, int64_t, node_ptr, StartTime, M)
    DECL_ACCESSOR_PROXY_S(SetEndTime, GetEndTime, int64_t, node_ptr, EndTime, M)
    DECL_ACCESSOR_PROXY_S(SetComputeCost, GetComputeCost, int64_t, node_ptr, ComputeCost, M)
    DECL_ACCESSOR_PROXY_S(SetTemporaryMemory, GetTemporaryMemory, int64_t, node_ptr, TemporaryMemory, M)
    DECL_ACCESSOR_PROXY_S(SetPersistentMemory, GetPersistentMemory, int64_t, node_ptr, PersistentMemory, M)
    DECL_ACCESSOR_PROXY_S(SetInputMemory, GetInputMemory, int64_t, node_ptr, InputMemory, M)
    DECL_ACCESSOR_PROXY_S(SetOutputMemory, GetOutputMemory, int64_t, node_ptr, OutputMemory, M)
    DECL_GETTER_PROXY(GetInputPorts, node_ptr, InputPorts)
    DECL_GETTER_PROXY(GetOutputPorts, node_ptr, OutputPorts)
    void AddInput(const std::string& input) {
        node_ptr->AddInput(std::move(input));
    }
    void AddOutput(const std::string& output) {
        node_ptr->AddOutput(std::move(output));
    }
    size_t InputSize() {
        return node_ptr->InputSize();
    }
    size_t OutputSize() {
        return node_ptr->OutputSize();
    }
    std::vector<int> InputIndexes() {
        return node_ptr->InputPorts() | ranges::views::transform([](const auto& p) { return p.index; })
               | ranges::to_vector;
    }
    std::vector<int> OutputIndexes() {
        return node_ptr->OutputPorts() | ranges::views::transform([](const auto& p) { return p.index; })
               | ranges::to_vector;
    }
    void AddInputPort(const std::string& input_node, int input_index, int index, DataType dtype, const shape_t& shape) {
        auto r = node_ptr->AddInputPort(input_node, input_index, index, dtype, shape);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
    }
    void AddOutputPort(DataType dtype, const shape_t& shape, int index) {
        auto r = node_ptr->AddOutputPort(dtype, shape, index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
    }
    void AddOutputPortWithBuffer(DataType dtype, const shape_t& shape, int index, const pybind11::array& array) {
        auto len = static_cast<size_t>(array.nbytes());
        void* ptr = std::malloc(len);
        auto info = array.request();
        memcpy(ptr, info.ptr, len);
        auto r = node_ptr->AddOutputPort(dtype, shape, index, ptr, len);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
    }
    StrAndInt GetInputRef(int index) {
        auto r = node_ptr->InputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value().entity.Ref();
    }
    StrAndInt GetInputName(int index) {
        auto r = node_ptr->InputName(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value();
    }
    StrAndInt GetOutputName(int index) {
        auto r = node_ptr->OutputName(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value();
    }
    DataType GetInputType(int index) {
        auto r = node_ptr->InputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value().entity.tensor.dtype;
    }
    shape_t GetInputShape(int index) {
        auto r = node_ptr->InputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value().entity.tensor.shape;
    }
    DataType GetOutputType(int index) {
        auto r = node_ptr->OutputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value().entity.dtype;
    }
    pybind11::array GetOutputValue(int index) {
        auto r = node_ptr->OutputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        auto& t = r.value().entity;
        if (t.data == nullptr) {
            return pybind11::none();
        }
        void* ptr = std::malloc(t.length);
        memcpy(ptr, t.data, t.length);
        // pass data by bytes
        pybind11::buffer_info info(ptr, 1, pybind11::format_descriptor<uint8_t>::format(), t.length);

        return pybind11::array(info);
    }
    shape_t GetOuputShape(int index) {
        auto r = node_ptr->OutputPort(index);
        if (r.has_error()) {
            throw std::invalid_argument(r.error().text);
        }
        return r.value().entity.shape;
    }
    std::string ToString() {
        return fmt::to_string(node_ptr);
    }
};
class Graph {
  private:
    geesibling::Graph* graph_ptr;

  public:
    explicit Graph(geesibling::Graph* graph) {
        graph_ptr = graph;
    }
    Graph(Graph&& graph) noexcept {
        graph_ptr = graph.graph_ptr;
    }
    Graph() {
        graph_ptr = new geesibling::Graph();
    }
    ~Graph() {
        delete graph_ptr;
    }
    geesibling::Graph* GraphPtr() {
        return graph_ptr;
    }
    void AddNode(Node& node) {
        graph_ptr->AddNode(node.NodePtr());
    }
    void AddNode(int at, Node& node) {
        graph_ptr->AddNode(at, node.NodePtr());
    }

    pybind11::object GetNode(int at, bool error = false) {
        auto r = graph_ptr->GetNode(at);
        if (r.has_error()) {
            if (error) {
                throw std::runtime_error(r.error().text);
            }
            return pybind11::none();
        }
        return pybind11::cast(new Node(r.value()));
    }
    pybind11::object GetNode(const std::string& name, bool error = false) {
        auto r = graph_ptr->GetNode(name);
        if (r.has_error()) {
            if (error) {
                throw std::runtime_error(r.error().text);
            }
            return pybind11::none();
        }
        return pybind11::cast(new Node(r.value()));
    }
    void AddReturn(const StrAndInt& r) {
        graph_ptr->AddReturn(r);
    }
    void ClearOutput() {
        graph_ptr->ClearReturn();
    }
    void ClearNode() {
        graph_ptr->ClearNode();
    }
    std::string ToString() {
        return fmt::to_string(*graph_ptr);
    }
};
class SubGraph {
  private:
    std::shared_ptr<geesibling::SubGraph> subgraph_ptr;

  public:
    explicit SubGraph(geesibling::SubGraph& subgraph) {
        subgraph_ptr = std::make_shared<geesibling::SubGraph>(subgraph);
    }
    explicit SubGraph(geesibling::SubGraphPtr subgraph) {
        subgraph_ptr = std::move(subgraph);
    }
    std::shared_ptr<geesibling::SubGraph>& SubGraphPtr() {
        return this->subgraph_ptr;
    }
    pybind11::object GetNode(int at, bool error = false) {
        auto r = subgraph_ptr->GetNode(at);
        if (r.has_error()) {
            if (error) {
                throw std::runtime_error(r.error().text);
            }
            return pybind11::none();
        }
        return pybind11::cast(new Node(r.value()));
    }
    pybind11::object GetNode(const std::string& name, bool error = false) {
        auto r = subgraph_ptr->GetNode(name);
        if (r.has_error()) {
            if (error) {
                throw std::runtime_error(r.error().text);
            }
            return pybind11::none();
        }
        return pybind11::cast(new Node(r.value()));
    }
    int GetNodesNum() {
        return subgraph_ptr->GetNodesNum();
    }
    DECL_GETTER_PROXY(GetInputs, subgraph_ptr, GetInputs)
    DECL_GETTER_PROXY(GetOutputs, subgraph_ptr, GetOutputs)

    std::string ToString() {
        return fmt::format("{:s}", fmt_shared(subgraph_ptr));
    }
    ~SubGraph() = default;
};

};  // namespace geesibling::py

namespace py = pybind11;
using PyNode = geesibling::py::Node;
using PyGraph = geesibling::py::Graph;
using PySubGraph = geesibling::py::SubGraph;
std::map<std::string, std::string> GetDeviceMapFromCostNodes(std::vector<geesibling::CostNode>& nodes) {
    return nodes | ranges::views::transform([](auto& a) { return std::make_pair(a.GetName(), a.GetDevice()); })
           | ranges::to<std::map<std::string, std::string>>();
}
PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
    m.doc() = R"pbdoc(
        python graph
        -----------------------
        .. currentmodule:: _graph
    )pbdoc";

    py::enum_<geesibling::DataType>(m, "DataType")
        .value("BOOL", geesibling::DataType::BOOL)
        .value("U8", geesibling::DataType::U8)
        .value("I8", geesibling::DataType::I8)
        .value("U16", geesibling::DataType::U16)
        .value("I16", geesibling::DataType::I16)
        .value("U32", geesibling::DataType::U32)
        .value("I32", geesibling::DataType::I32)
        .value("U64", geesibling::DataType::U64)
        .value("I64", geesibling::DataType::I64)
        .value("F8E4M3FN", geesibling::DataType::F8E4M3FN)
        .value("F8E5M2", geesibling::DataType::F8E5M2)
        .value("BF16", geesibling::DataType::BF16)
        .value("F16", geesibling::DataType::F16)
        .value("F32", geesibling::DataType::F32)
        .value("F64", geesibling::DataType::F64)
        .value("Other", geesibling::DataType::Other);

    py::class_<PyNode>(m, "Node")
        .def(py::init())
        .def(py::init([](const PyNode& node) { return PyNode(node); }))
        .def(py::init([](const std::string& name, const std::string& op) {
            auto n = std::make_unique<PyNode>();
            n->SetName(std::move(name));
            n->SetOp(std::move(op));
            return n;
        }))
        .def_property("name", &PyNode::GetName, &PyNode::SetName)
        .def_property("op", &PyNode::GetOp, &PyNode::SetOp)
        .def_property("device", &PyNode::GetDevice, &PyNode::SetDevice)
        .def_property("inputs", &PyNode::GetInputs, &PyNode::SetInputs)
        .def_property("outputs", &PyNode::GetOutputs, &PyNode::SetOutputs)
        .def_property("attrs", &PyNode::GetAttrs, &PyNode::SetAttrs)
        .def_property("start_time", &PyNode::GetStartTime, &PyNode::SetStartTime)
        .def_property("end_time", &PyNode::GetEndTime, &PyNode::SetEndTime)
        .def_property("compute_cost", &PyNode::GetComputeCost, &PyNode::SetComputeCost)
        .def_property("temporary_memory", &PyNode::GetTemporaryMemory, &PyNode::SetTemporaryMemory)
        .def_property("persistent_memory", &PyNode::GetPersistentMemory, &PyNode::SetPersistentMemory)
        .def_property("input_memory", &PyNode::GetInputMemory, &PyNode::SetInputMemory)
        .def_property("output_memory", &PyNode::GetOutputMemory, &PyNode::SetOutputMemory)
        .def("input_indexes", &PyNode::InputIndexes)
        .def("output_indexes", &PyNode::OutputIndexes)
        .def("input_ref", &PyNode::GetInputRef)
        .def("input_name", &PyNode::GetInputName)
        .def("input_type", &PyNode::GetInputType)
        .def("input_shape", &PyNode::GetInputShape)
        .def("output_type", &PyNode::GetOutputType)
        .def("output_shape", &PyNode::GetOuputShape)
        .def("output_name", &PyNode::GetOutputName)
        .def("output_value", &PyNode::GetOutputValue)
        .def("add_input", &PyNode::AddInput)
        .def("add_output", &PyNode::AddOutput)
        .def("add_inputport", &PyNode::AddInputPort)
        .def("add_outputport", &PyNode::AddOutputPort)
        .def("add_outputport", &PyNode::AddOutputPortWithBuffer)
        .def("__repr__", &PyNode::ToString)
        .def("__str__", &PyNode::ToString);

    py::class_<PyGraph>(m, "Graph")
        .def(py::init())
        .def("add_node", py::overload_cast<PyNode&>(&PyGraph::AddNode))
        .def("add_node", py::overload_cast<int, PyNode&>(&PyGraph::AddNode))
        .def("get_node", py::overload_cast<int, bool>(&PyGraph::GetNode), py::arg("index"), py::arg("error") = false)
        .def("get_node", py::overload_cast<const std::string&, bool>(&PyGraph::GetNode), py::arg("name"),
             py::arg("error") = false)
        .def("add_return", &PyGraph::AddReturn)
        .def_property_readonly("node_num", [](PyGraph& g) { return g.GraphPtr()->GetNodesNum(); })
        .def_property_readonly("returns", [](PyGraph& g) { return g.GraphPtr()->Returns(); })
        .def_property_readonly("nodes",
                               [](PyGraph& g) {
                                   return g.GraphPtr()->Nodes()
                                          | ranges::views::transform([](auto& node) { return PyNode(node); })
                                          | ranges::to_vector;
                               })
        .def("__repr__", &PyGraph::ToString)
        .def("__str__", &PyGraph::ToString);

    py::class_<PySubGraph>(m, "SubGraph")
        .def(py::init([](const PySubGraph& subgraph) { return PySubGraph(subgraph); }))
        .def_property_readonly("nodes_num", &PySubGraph::GetNodesNum)
        .def("get_node", py::overload_cast<int, bool>(&PySubGraph::GetNode), py::arg("index"), py::arg("error") = false)
        .def("get_node", py::overload_cast<const std::string&, bool>(&PySubGraph::GetNode), py::arg("name"),
             py::arg("error") = false)
        .def_property_readonly("inputs", &PySubGraph::GetInputs)
        .def_property_readonly("outputs", &PySubGraph::GetOutputs)
        .def_property_readonly(
            "input_graphs",
            [](PySubGraph& g) {
                auto& a = g.SubGraphPtr()->GetInputGraphs();
                return a | ranges::views::transform([](geesibling::SubGraphPtr& i) { return PySubGraph(i); })
                       | ranges::to_vector;
            })
        .def_property_readonly("output_graphs",
                               [](PySubGraph& g) {
                                   auto& a = g.SubGraphPtr()->GetOutputGraphs();
                                   return a | ranges::views::transform([](auto& i) { return PySubGraph(i); })
                                          | ranges::to_vector;
                               })
        .def("add_return", [](PySubGraph& g, const StrAndInt& r) { return g.SubGraphPtr()->AddReturn(r); })
        .def_property_readonly("returns", [](PySubGraph& g) { return g.SubGraphPtr()->Returns(); })
        .def_property_readonly("nodes",
                               [](PySubGraph& g) {
                                   return g.SubGraphPtr()->Nodes()
                                          | ranges::views::transform([](auto& node) { return PyNode(node); })
                                          | ranges::to_vector;
                               })
        .def("__str__", &PySubGraph::ToString)
        .def("__repr__", [](PySubGraph& g) { return fmt::format("{:s}", fmt_shared(g.SubGraphPtr())); })
        .def("__hash__", [](PySubGraph& g) { return std::hash<geesibling::SubGraphPtr>()(g.SubGraphPtr()); })
        .def("__eq__",
             [](PySubGraph& g, PySubGraph& other) { return g.SubGraphPtr().get() == other.SubGraphPtr().get(); });
    using PyBlock = geesibling::Block;
    py::class_<PyBlock>(m, "Block")
        .def(py::init([](const std::string& device) { return geesibling::Block(device); }))
        .def(py::init([](PySubGraph& graph) { return geesibling::Block(graph.SubGraphPtr()); }))
        .def_property_readonly("id", [](PyBlock& b) { return b.Id(); })
        .def_property_readonly("device", [](PyBlock& b) { return b.Device(); })
        .def("__str__", [](PyBlock& b) { return fmt::to_string(b); })
        .def("__repr__", [](PyBlock& b) { return fmt::format("{:s}", b); })
        .def_property_readonly("graph", [](PyBlock& b) { return PySubGraph(b.GetSubGraph()); })
        .def("add_inputport", [](PyBlock& b, int64_t blockId, int index, geesibling::DataType dtype,
                                 geesibling::shape_t& shape) { b.AddInputPort(blockId, index, dtype, shape); })
        .def("add_outputport",
             [](PyBlock& b, geesibling::DataType dtype, geesibling::shape_t& shape) { b.AddOutputPort(dtype, shape); })
        .def_property_readonly("inputports",
                               [](PyBlock& block) {
                                   return block.Inputs() | ranges::views::transform([](auto& i) {
                                              return std::tuple(i.index, i.entity.source, i.entity.index,
                                                                i.entity.tensor.dtype, i.entity.tensor.shape);
                                          })
                                          | ranges::to_vector;
                               })
        .def_property_readonly("outputports",
                               [](PyBlock& block) {
                                   return block.Outputs() | ranges::views::transform([](auto& i) {
                                              return std::tuple(i.index, i.entity.dtype, i.entity.shape);
                                          })
                                          | ranges::to_vector;
                               })
        .def_property_readonly("inputports_size", [](PyBlock& block) { return block.Inputs().size(); })
        .def_property_readonly("outputports_size", [](PyBlock& block) { return block.Outputs().size(); })
        .def("__hash__", [](PyBlock& block) { return block.Id(); })
        .def("__eq__", [](PyBlock& block, PyBlock& other) { return block.Id() == other.Id(); });
    py::enum_<geesibling::DeviceType>(m, "DeviceType")
        .value("cpu", geesibling::DeviceType::Cpu)
        .value("gpu", geesibling::DeviceType::NVGpu)
        .value("rocm", geesibling::DeviceType::AMDGpu)
        .value("ascend", geesibling::DeviceType::Ascend);
    using PyDevice = geesibling::Device;
    py::class_<PyDevice>(m, "Device")
        .def(py::init(
            [](geesibling::DeviceType type, std::string name, int64_t memory, int64_t free_memory,
               int64_t execute_time) { return PyDevice(type, std::move(name), memory, free_memory, execute_time); }))
        .def("__str__", [](PyDevice& b) { return fmt::to_string(b); })
        .def("__repr__", [](PyDevice& b) { return fmt::to_string(b); });

    m.def("divide_graph", [](PyGraph& graph) {
        auto r = geesibling::DivideGraph(*graph.GraphPtr());
        if (r.has_error()) {
            throw std::runtime_error(r.error().text);
        }
        return r.value() | ranges::views::values | ranges::views::transform([](auto& g) { return PySubGraph(g); })
               | ranges::to_vector;
    });
    m.def("search_policy",
          [](PyGraph& graph, std::vector<PyDevice> devices, const std::string& policy) -> pybind11::object {
              SPDLOG_DEBUG("devices:{}", devices);
              SPDLOG_DEBUG("policy:{}", policy);
              if (policy == "fddps") {
                  geesibling::CostGraph cost_graph = geesibling::ConvertGraphToCostGraph(*graph.GraphPtr());
                  geesibling::FDDPSAlgorithm fddps_algorithm(cost_graph, std::move(devices));
                  auto r = fddps_algorithm.Placement();
                  if (r.has_error()) {
                      SPDLOG_ERROR("call fddps error. {}", r.error().text);
                      return pybind11::none();
                  }
                  auto device_map = GetDeviceMapFromCostNodes(r.value());
                  return pybind11::cast(device_map);
              }
              if (policy == "sgp") {
                  geesibling::Partition partition(*graph.GraphPtr(), devices.size(), devices, 0.6, 1);
                  auto& device_map = partition.op_group;
                  SPDLOG_DEBUG("sgp result:{}", device_map);
                  return pybind11::cast(device_map);
              }
              return pybind11::none();
          });
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
