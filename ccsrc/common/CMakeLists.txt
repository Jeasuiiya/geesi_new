# add_subdirectory(include)
# set(GEESIBLING_THIRD_COMMON_LIBS "Result::Result;fmt::fmt;range-v3")
add_header_only_library(CommonHeaders ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_INSTALL_INCLUDEDIR}/common
                        DEPENDS ThirdCommon)
