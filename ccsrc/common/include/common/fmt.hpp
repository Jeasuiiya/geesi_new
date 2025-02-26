#ifndef GEESIBLING_FMT_HPP
#define GEESIBLING_FMT_HPP
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "fmt/format.h"
#include "fmt/ranges.h"

template <typename T>
using ostreamable_t =
    std::enable_if_t<fmt::has_formatter<T, fmt::format_context>::value && !std::is_convertible_v<T, std::string>, int>;
template <typename T>
using formatable_t = std::enable_if_t<fmt::has_formatter<T, fmt::format_context>::value, int>;

#if __cplusplus >= 202002L
template <typename T>
concept formatable = fmt::has_formatter<T, fmt::format_context>::value;

// clang-format off
template <typename T>
concept ostreamable = formatable<T> && !std::is_convertible_v<T, std::string>;
// clang-format on
#endif

#if __cplusplus >= 202002L
template <ostreamable T>
#else
template <typename T, ostreamable_t<T> = 0>
#endif
inline std::ostream& operator<<(std::ostream& os, const T& t) {
    os << fmt::to_string(t);
    return os;
}

// NOLINTBEGIN
template <typename T>
class fmt_unique {
    friend struct fmt::formatter<fmt_unique<T>>;
    const std::unique_ptr<T>& ptr;

  public:
    explicit fmt_unique(const std::unique_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_shared {
    friend struct fmt::formatter<fmt_shared<T>>;
    const std::shared_ptr<T>& ptr;

  public:
    explicit fmt_shared(const std::shared_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_weak {
    friend struct fmt::formatter<fmt_weak<T>>;
    const std::weak_ptr<T> ptr;

  public:
    explicit fmt_weak(const std::weak_ptr<T>& ptr) : ptr(ptr) {}
};
struct ShortFormat;
template <>
struct fmt::formatter<ShortFormat> {
    // f: full
    // s: short
    char presentation = 'f';
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        const auto* it = ctx.begin();
        const auto* end = ctx.end();
        if (it != end && (*it == 's' || *it == 'f')) {
            presentation = *it++;
        }

        // Check if reached the end of the range:
        if (it != end && *it != '}') {
            throw format_error("invalid format");
        }

        // Return an iterator past the end of the parsed range:
        return it;
    }
};
template <typename T>
struct fmt::formatter<std::shared_ptr<T>> : public fmt::formatter<T> {
    template <typename FormatContext, class = formatable_t<T>>
    auto format(const std::shared_ptr<T>& ptr, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<T>::format(*ptr.get(), ctx);
    }
};
template <typename T>
struct fmt::formatter<fmt_shared<T>> : public fmt::formatter<T> {
    template <typename FormatContext, class = formatable_t<T>>
    auto format(const fmt_shared<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<T>::format(*proxy.ptr, ctx);
    }
};
template <typename T>
struct fmt::formatter<fmt_weak<T>> : public fmt::formatter<fmt_shared<T>> {
    template <typename FormatContext, class = formatable_t<T>>
    auto format(const fmt_weak<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<fmt_shared<T>>::format(fmt_shared(proxy.ptr.lock()), ctx);
    }
};
template <typename T>
struct fmt::formatter<fmt_unique<T>> : public fmt::formatter<T> {
    template <typename FormatContext, class = formatable_t<T>>
    auto format(const fmt_unique<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<T>::format(*proxy.ptr, ctx);
    }
};
// NOLINTEND

#endif
