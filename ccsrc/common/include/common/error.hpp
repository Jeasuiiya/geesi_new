#ifndef GEESIBLING_ERROR_HPP
#define GEESIBLING_ERROR_HPP
#include <string>
enum class Kind { Unknown, Invalid, Unimplemented, Internal };
struct Error {
    Error(Kind kind, std::string text) noexcept {
        this->kind = kind;
        this->text = std::move(text);
    }
    Error() noexcept {
        this->kind = Kind::Unknown;
        this->text = "Unknown Error";
    }
    Kind kind;
    std::string text;
};
#endif
