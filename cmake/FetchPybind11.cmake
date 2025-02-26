# require CPM
include(CPM)
include(CMakeParseArguments)
# cmake-lint: disable=C0103
# fetch_pybind11
function(fetch_pybind11)
  if(NOT TARGET pybind11::pybind11)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
    CPMAddPackage(
      NAME pybind11
      GITHUB_REPOSITORY pybind/pybind11
      GIT_TAG v2.10.3
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_pybind11)

# add_pybind11_module <target> <output_path> [SRCS srcs...] [DEPENDS targets...]
function(add_pybind11_module target_name output_path)
  cmake_parse_arguments(_ARG "" "" "SRCS;DEPENDS" ${ARGN})

  pybind11_add_module(${target_name} ${_ARG_SRCS})
  target_compile_definitions(${target_name} PRIVATE VERSION_INFO=${PROJECT_VERSION}
                                                    PYBIND11_CURRENT_MODULE_NAME=${target_name})
  target_link_libraries(${target_name} PRIVATE ${_ARG_DEPENDS})
  add_dependencies(pymodule ${target_name})
  set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${output_path}")
endfunction(add_pybind11_module)

add_custom_target(pymodule COMMENT "pymodule")
