include(GoogleTest)
function(add_cc_gtest targe_name)
    cmake_parse_arguments(_ARG "" "" "SRCS;DEPENDS;DEFINITIONS" ${ARGN})
    add_executable(
        ${targe_name}
        ${_ARG_SRCS}
    )
    target_link_libraries(
        ${targe_name}
        GTest::gtest_main
        ${_ARG_DEPENDS}
    )
    target_compile_options(${targe_name} PRIVATE -fno-access-control)
    target_compile_definitions(${targe_name} PRIVATE ${_ARG_DEFINITIONS})
    gtest_discover_tests(${targe_name})
endfunction(add_cc_gtest)

add_subdirectory(ut)
