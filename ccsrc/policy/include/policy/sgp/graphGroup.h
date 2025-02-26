#include <cfloat>
#include <set>
#include <string>
#include <vector>

class SgpEdge {
  public:
    std::string pre_op;
    std::string succ_op;
    std::int64_t comm_cost;
    std::int64_t choosed_times;

    SgpEdge(const std::string& pre, const std::string& succ, std::int64_t comm) {
        this->pre_op = pre;
        this->succ_op = succ;
        this->comm_cost = comm;
        this->choosed_times = 0;
    }
};

class GraphGroup {
  public:
    std::int64_t id;
    std::set<std::string> op_member;
    std::int64_t op_cost;
    std::int64_t comm_cost_inner;
    std::int64_t comm_cost_outer;
    std::vector<SgpEdge> critical_edges;
    std::int64_t balance_cost;
    float balance_factor;

    GraphGroup(std::int64_t id_, std::int64_t balance_cost_) {
        this->id = id_;
        this->op_cost = 0;
        this->comm_cost_inner = 0;
        this->comm_cost_outer = 0;
        this->balance_cost = balance_cost_;
        this->balance_factor = 1;
    }

    [[nodiscard]] bool Contains(const std::string& op) const {
        return this->op_member.count(op) == 1;
    }

    void AddOp(const std::string& op) {
        this->op_member.insert(op);
    }

    void RemoveOp(const std::string& op) {
        this->op_member.erase(op);
    }

    void AddCriticalEdge(std::string& pre, std::string& succ, std::int64_t comm) {
        this->critical_edges.emplace_back(pre, succ, comm);
    }

    void SetBalanceCost(std::int64_t balance_cost_) {
        this->balance_cost = balance_cost_;
    }

    void UpdateBalanceFactor() {
        if (this->balance_cost != 0) {
            this->balance_factor = static_cast<double>(this->GetTotalCost()) / this->balance_cost;
        }
    }

    [[nodiscard]] std::set<std::string> GetOpMember() const {
        return this->op_member;
    }

    [[nodiscard]] std::int64_t GetTotalCost() const {
        return this->op_cost + this->comm_cost_inner;
    }
};
