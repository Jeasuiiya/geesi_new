#pragma once
#include <functional>
#ifndef GEESIBLING_COMMON_TRAITS_H
#define GEESIBLING_COMMON_TRAITS_H
// NOLINTBEGIN
template <class T>
struct function_trait;

template <class R, class... Args>
struct function_trait<std::function<R(Args...)>> {
    static const size_t nargs = sizeof...(Args);
    using result_type = R;
    template <size_t i>
    struct arg {
        using type = std::tuple_element_t<i, std::tuple<Args...>>;
    };
};

template <class R, class... Args>
struct function_trait<std::function<R (*)(Args...)>> {
    static const size_t nargs = sizeof...(Args);
    using result_type = R;
    template <size_t i>
    struct arg {
        using type = std::tuple_element_t<i, std::tuple<Args...>>;
    };
};

template <class C, class R, class... Args>
struct function_trait<std::function<R (C::*)(Args...)>> {
    static const size_t nargs = sizeof...(Args);
    using result_type = R;
    template <size_t i>
    struct arg {
        using type = std::tuple_element_t<i, std::tuple<Args...>>;
    };
};

// NOLINTEND
#endif
