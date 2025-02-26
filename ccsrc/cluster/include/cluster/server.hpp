#pragma once
#ifndef GEESIBLING_CLUSTER_SERVER_H
#define GEESIBLING_CLUSTER_SERVER_H

#include <string>
#include <vector>

#include "common/fmt.hpp"
#include "common/util.hpp"

namespace geesibling {
enum class DeviceStatus {
    Using,  // 使用
    Idle    // 闲置
};
enum class DeviceType { Cpu, NVGpu, AMDGpu, Ascend };
static inline DeviceType DeviceTypeFrom(const std::string& s) {
    if (s == "cpu" || s == "CPU") {
        return DeviceType::Cpu;
    }
    if (s == "gpu" || s == "GPU") {
        return DeviceType::NVGpu;
    }
    if (s == "rocm" || s == "AMDGPU") {
        return DeviceType::AMDGpu;
    }
    if (s == "ascend" || s == "Ascend") {
        return DeviceType::Ascend;
    }
    return DeviceType::Cpu;
}
class Device {
    friend struct fmt::formatter<geesibling::Device>;

  private:
    // DeviceStatus status;   // 设备使用状态
    DeviceType type;       // 设备类型
    std::string name;      // 设备逻辑名称
    int64_t memory;        // 设备总内存
    int64_t free_memory;   // 已使用内存
    int64_t execute_time;  // 设备的执行时间

  public:
    Device() : type(geesibling::DeviceType::Cpu), memory(0), free_memory(0), execute_time(0) {}
    Device(DeviceType _type, std::string _name, int64_t _memory, int64_t _free_memory, int64_t _execute_time)
        :  // : status(std::move(_status)),
          type(std::move(_type)),
          name(std::move(_name)),
          memory(std::move(_memory)),
          free_memory(std::move(_free_memory)),
          execute_time(std::move(_execute_time)){};
    virtual ~Device() = default;

    // DECL_ACCESSOR(GetStatus, SetStatus, status, M)
    DECL_ACCESSOR(GetType, SetType, type, M)
    DECL_ACCESSOR(GetName, SetName, name, M)
    DECL_ACCESSOR(GetMemory, SetMemory, memory, M)
    DECL_ACCESSOR(GetFreeMemory, SetFreeMemory, free_memory, M)
    DECL_ACCESSOR(GetExecuteTime, SetExecuteTime, execute_time, M)
};

// enum class LinkType { PCIE, NVLink, Eth };
// class Link {
//     LinkType type;                 // 链接类型
//     std::vector<Device*> devices;  // 可用设备
//     long totalBandwidth;           // 共享总带宽
// };

// class Server {
//     DeviceStatus status;          // 设备使用状态
//     std::string name;             // 服务器逻辑名称
//     std::vector<Device> devices;  // 计算设备
//     long usedMemory;              // 已使用内存
//     long totalMemory;             // 总内存
//     std::vector<Link> links;      // 链接
// };

};  // namespace geesibling

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<geesibling::Device> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::Device& d, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Device(name={}, memory={}, free_memory={}, execute_time={})", d.name,
                              d.memory, d.free_memory, d.execute_time);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif
