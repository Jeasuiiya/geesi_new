#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/fmt.hpp"
#include "common/id.hpp"
#include "gtest/gtest.h"
#include "range/v3/all.hpp"
// NOLINTBEGIN(readability-identifier-naming)
struct A {};
template <>
struct fmt::formatter<A> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const A&, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "A()");
    }
};
// NOLINTEND(readability-identifier-naming)

namespace geesibling {
// Demonstrate some basic assertions.
TEST(TestCommon, FmtRaiiPtr) {
    EXPECT_NO_THROW(std::cout << std::make_shared<std::string>("shared string") << std::endl);
    EXPECT_NO_THROW(std::cout << fmt_unique(std::make_unique<std::string>("unique string")) << std::endl);
    EXPECT_NO_THROW(std::cout << fmt_unique(std::make_unique<A>()) << std::endl);

    // vector unique_ptr
    auto a = std::make_unique<A>();
    std::vector<std::unique_ptr<A>> vn;
    vn.push_back(std::move(a));
    auto vi = vn | ranges::views::transform([](const auto& ptr) { return fmt_unique(ptr); });
    EXPECT_NO_THROW(std::cout << vi << std::endl);

    // map unique_ptr
    std::multimap<std::string, std::unique_ptr<A>> m;
    m.insert({"1", std::make_unique<A>()});
    m.insert({"1", std::make_unique<A>()});
    auto mv = m | ranges::views::values | ranges::views::transform([](const auto& i) { return fmt_unique(i); });
    auto mr = ranges::views::zip(ranges::views::keys(m), mv) | ranges::to<std::multimap<std::string, fmt_unique<A>>>();
    EXPECT_NO_THROW(std::cout << mr << std::endl);
    auto g1 = IDGenerator.Gen();
    auto g2 = IDGenerator.Gen();
    EXPECT_NE(g1, g2);
}
}  // namespace geesibling
