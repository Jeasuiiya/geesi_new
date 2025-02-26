#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "iostream"
#include "map"
#include "string"
#include "vector"

namespace geesibling {
// 定义栈
struct Stack {
    std::vector<std::string> Sacklist;
    std::int64_t top = -1;
};

// 入栈操作
static void Pushs(Stack& s, const std::string& key) {
    s.top++;
    s.Sacklist.push_back(key);
}

// 出栈操作
static std::string Pops(Stack& s) {
    if (s.top == -1) {
        return "ERROR";
    }
    std::string temp = s.Sacklist[s.top];
    s.Sacklist.pop_back();
    s.top--;
    return temp;
}

static std::vector<std::string> TopoLogical(Graph& graph) {
    std::vector<std::string> topo_list;
    // 记录每个节点的入度
    std::map<std::string, std::int64_t> in_degree;
    const std::vector<std::shared_ptr<NodeBase>>& graph_nodes = graph.Nodes();
    for (const auto& node : graph_nodes) {
        in_degree.insert(std::pair<std::string, std::int64_t>((*node).Name(), (*node).Inputs().size()));
    }

    Stack s;

    // 先将所有入度为0的节点入栈
    for (const auto& it : in_degree) {
        if (it.second == 0) {
            Pushs(s, it.first);
        }
    }

    while (s.top != -1) {
        std::string vx = Pops(s);
        if (vx == "ERROR") {
            break;
        }
        topo_list.push_back(vx);
        NodeBase it = *graph.GetNode(vx).value();
        for (const auto& jt : it.Outputs()) {
            in_degree[jt]--;
            if (in_degree[jt] == 0) {
                Pushs(s, jt);
            }
        }
    }

    return topo_list;
}
}  // namespace geesibling
