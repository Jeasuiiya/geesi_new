#pragma once
#include <functional>
#ifndef GEESIBLING_COMMON_ID_HPP
#define GEESIBLING_COMMON_ID_HPP
#include <cstdint>
#include <ctime>

namespace geesibling {

namespace details {

class IDGenerator {
  private:
    // reserve 0
    int64_t current{1};

  public:
    int64_t Gen() {
        return current++;
    }
    IDGenerator() = default;
    IDGenerator(const IDGenerator&) = delete;
    IDGenerator(IDGenerator&&) = delete;
};

}  // namespace details
// NOLINTBEGIN
// not thread safety
static details::IDGenerator IDGenerator;
// NOLINTEND

}  // namespace geesibling

#endif
