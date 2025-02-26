#ifndef GEESIBLING_UTIL_H
#define GEESIBLING_UTIL_H
#include <type_traits>

#include "traits.hpp"
#define ALL(...) __VA_ARGS__

#ifndef DECL_SETTER
#define DECL_SETTER(SetterName, fieldName, MOVE_OR_COPY) DECL_SETTER_##MOVE_OR_COPY(SetterName, fieldName)
#endif

#ifndef DECL_SETTER_M
#define DECL_SETTER_M(SetterName, fieldName)                                                                \
    /* NOLINTNEXTLINE */                                                                                    \
    inline void SetterName(std::add_lvalue_reference_t<std::add_const_t<decltype(fieldName)>>(fieldName)) { \
        this->fieldName = std::move(fieldName);                                                             \
    }
#endif
#ifndef DECL_SETTER_C
#define DECL_SETTER_C(SetterName, fieldName)                                                                \
    /* NOLINTNEXTLINE */                                                                                    \
    inline void SetterName(std::add_lvalue_reference_t<std::add_const_t<decltype(fieldName)>>(fieldName)) { \
        this->fieldName = fieldName;                                                                        \
    }
#endif

#ifndef DECL_GETTER
#define DECL_GETTER(GetterName, fieldName)                                       \
    inline auto GetterName()->std::add_lvalue_reference_t<decltype(fieldName)> { \
        return fieldName;                                                        \
    }
#endif

#ifndef DECL_ACCESSOR
#define DECL_ACCESSOR(GetterName, SetterName, fName, MOVE) \
    DECL_SETTER(SetterName, fName, MOVE)                   \
    DECL_GETTER(GetterName, fName)
#endif

#ifndef DECL_SETTER_PROXY
#define DECL_SETTER_PROXY(SetterName, Type, Proxy, ProxySetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY_##MOVE_OR_COPY(SetterName, ALL(Type), Proxy, ProxySetterName)
#endif

#ifndef DECL_SETTER_PROXY_M
#define DECL_SETTER_PROXY_M(SetterName, Type, Proxy, ProxySetterName) \
    /* NOLINTNEXTLINE */                                              \
    inline void SetterName(const Type& _set_value) {                  \
        /* NOLINTNEXTLINE */                                          \
        this->Proxy->ProxySetterName(std::move(_set_value));          \
    }
#endif

#ifndef DECL_SETTER_PROXY_C
#define DECL_SETTER_PROXY_C(SetterName, Type, Proxy, ProxySetterName) \
    /* NOLINTNEXTLINE */                                              \
    inline void SetterName(const Type& _set_value) {                  \
        /* NOLINTNEXTLINE */                                          \
        this->Proxy->ProxySetterName(_set_value);                     \
    }
#endif

#ifndef DECL_GETTER_PROXY
#define DECL_GETTER_PROXY(GetterName, Proxy, ProxyGetterName)                                         \
    inline auto GetterName()->std::add_lvalue_reference_t<decltype(this->Proxy->ProxyGetterName())> { \
        return this->Proxy->ProxyGetterName();                                                        \
    }
#endif

#ifndef DECL_ACCESSOR_PROXY
#define DECL_ACCESSOR_PROXY(SetterName, GetterName, Type, Proxy, ProxySetterName, ProxyGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterName, MOVE_OR_COPY)                               \
    DECL_GETTER_PROXY(GetterName, Proxy, ProxyGetterName)
#endif

// proxy setter is same to getter
#ifndef DECL_ACCESSOR_PROXY_S
#define DECL_ACCESSOR_PROXY_S(SetterName, GetterName, Type, Proxy, ProxySetterGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterName, ALL(Type), Proxy, ProxySetterGetterName, MOVE_OR_COPY)                \
    DECL_GETTER_PROXY(GetterName, Proxy, ProxySetterGetterName)
#endif

// proxy setter is same to getter
// setter is same to getter
#ifndef DECL_ACCESSOR_PROXY_SS
#define DECL_ACCESSOR_PROXY_SS(SetterGetterName, Type, Proxy, ProxySetterGetterName, MOVE_OR_COPY) \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, ProxySetterGetterName, MOVE_OR_COPY)     \
    DECL_GETTER_PROXY(SetterGetterName, Proxy, ProxySetterGetterName)
#endif

// all name is same
#ifndef DECL_ACCESSOR_PROXY_SSS
#define DECL_ACCESSOR_PROXY_SSS(SetterGetterName, Type, Proxy, MOVE_OR_COPY)              \
    DECL_SETTER_PROXY(SetterGetterName, ALL(Type), Proxy, SetterGetterName, MOVE_OR_COPY) \
    DECL_GETTER_PROXY(SetterGetterName, Proxy, SetterGetterName)
#endif

#endif /* ifndef GEESIBLING_UTIL_H */
