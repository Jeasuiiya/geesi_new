cmake_minimum_required(VERSION 3.22)

# provide args:
# GEESIBLING_BUILD_TEST
# GEESIBLING_ENABLE_BINDINGS_PYTHON
# PYTHON_EXECUTABLE
# CPM_SOURCE_CACHE
project(GeeSibling CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/third_party)

option(GEESIBLING_BUILD_TEST "build test" ON)

# cmake -DPYBIND11_PYTHON_VERSION=3.6 ..

# # Another method: cmake -DPYTHON_EXECUTABLE=/path/to/python ..

# # This often is a good way to get the current Python, works in environments:
# cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..
option(GEESIBLING_ENABLE_BINDINGS_PYTHON "python bindings" ON)

# begin config set(CMAKE_MESSAGE_LOG_LEVEL TRACE)
include(CPM)
include(HeaderOnlyLibrary)
include(FetchSpdlog)
include(FetchResult)
include(FetchFmt)
# provide ranges without c++20
include(FetchRange)
include(FetchLLVM)
include(FetchBoost)

# set(FETCHCONTENT_QUIET FALSE)

# fetch_llvm()
fetch_result()
fetch_fmt()
fetch_range()
fetch_spdlog()

# third libraries as common denpendencies
set(GEESIBLING_THIRD_COMMON_LIBS "")
list(APPEND GEESIBLING_THIRD_COMMON_LIBS Result::Result)
list(APPEND GEESIBLING_THIRD_COMMON_LIBS fmt::fmt)
list(APPEND GEESIBLING_THIRD_COMMON_LIBS range-v3)
list(APPEND GEESIBLING_THIRD_COMMON_LIBS spdlog::spdlog_header_only)

add_library(ThirdCommon INTERFACE)
target_link_libraries(ThirdCommon INTERFACE ${GEESIBLING_THIRD_COMMON_LIBS})

if(GEESIBLING_ENABLE_BINDINGS_PYTHON)
  include(FetchPybind11)
  fetch_pybind11()
  file(GLOB_RECURSE GEESIBLING_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/python/*.py)
  add_custom_target(
    python
    COMMENT "copy python sources"
    VERBATIM COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/python
            "${CMAKE_BINARY_DIR}/python"
    DEPENDS ${GEESIBLING_PYTHON_SRCS}
    SOURCES ${GEESIBLING_PYTHON_SRCS})
endif()

add_subdirectory(ccsrc)

if(GEESIBLING_BUILD_TEST)
  include(FetchGTest)
  set(INSTALL_GTEST OFF CACHE BOOL OFF FORCE)
  fetch_gtest()
  enable_testing()
  add_subdirectory(tests)
endif(GEESIBLING_BUILD_TEST)
