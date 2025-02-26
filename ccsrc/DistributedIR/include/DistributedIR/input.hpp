#pragma once

#ifndef GEESIBLING_IR_INPUT_H
#define GEESIBLING_IR_INPUT_H
#include <cstdint>

#include "DistributedIR/tensor.hpp"
namespace geesibling {

template <class T>
struct Input {
    static const int InvalidIndex = -1;
    // input source object
    T source;
    // output index of input source object
    int index{InvalidIndex};
    AbstractTensor tensor;

    Input(const T& node, int index, const AbstractTensor& tensor) : tensor(tensor) {
        this->source = node;
        this->index = index;
    }
    Input(T&& node, int index, AbstractTensor&& tensor) : tensor(std::move(tensor)) {
        this->source = std::move(node);
        this->index = index;
    }
    [[nodiscard]] std::pair<T, int> Ref() const {
        return std::make_pair(source, index);
    }
    bool operator==(const Input& input) {
        return Ref() == input.Ref() && tensor == input.tensor;
    }
};
using InputStr = Input<std::string>;

}  // namespace geesibling

// NOLINTBEGIN
template <class T>
struct fmt::formatter<geesibling::Input<T>> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const geesibling::Input<T>& input, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Input(ref={}, tensor={})", input.Ref(), input.tensor);
    }
};
// NOLINTEND

#endif
