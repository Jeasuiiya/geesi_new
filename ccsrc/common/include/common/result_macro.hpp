#ifndef GEESIBLING_COMMON_RESULT_HPP
#define GEESIBLING_COMMON_RESULT_HPP
#include "result.hpp"

#define RESULT_MACROS_CONCAT_NAME(x, y) RESULT_MACROS_CONCAT_IMPL(x, y)
#define RESULT_MACROS_CONCAT_IMPL(x, y) x##y

#define TRY_ASSIGN_IMPL(result, lhs, rexpr)            \
    /*NOLINTNEXTLINE*/                                 \
    auto result = (rexpr);                             \
    if ((result).has_error()) {                        \
        return cpp::fail(std::move((result)).error()); \
    }                                                  \
    /*NOLINTNEXTLINE*/                                 \
    lhs = std::move((result)).value()

#define TRY_ASSIGN(lhs, rexpr) TRY_ASSIGN_IMPL(RESULT_MACROS_CONCAT_NAME(_result_or_value, __COUNTER__), lhs, rexpr)

#define TRY_IMPL(result, rexpr)                        \
    /*NOLINTNEXTLINE*/                                 \
    auto result = (rexpr);                             \
    if ((result).has_error()) {                        \
        return cpp::fail(std::move((result)).error()); \
    }

#define TRY(rexpr) TRY_IMPL(RESULT_MACROS_CONCAT_NAME(_result_or_value, __COUNTER__), rexpr)

#endif
