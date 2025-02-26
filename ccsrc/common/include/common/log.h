#ifndef GEESIBLING_COMMON_LOG_H
#define GEESIBLING_COMMON_LOG_H
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable
#include "spdlog/spdlog.h"
namespace geesibling::log {
inline void Init() {
    spdlog::cfg::load_env_levels();
}
struct LogInit {
    LogInit() {
        Init();
    }
};
static LogInit log_init;
}  // namespace geesibling::log

#endif
