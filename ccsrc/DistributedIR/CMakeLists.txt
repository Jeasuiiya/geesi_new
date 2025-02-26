add_header_only_library(DistributedIRHeaders ${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CMAKE_INSTALL_INCLUDEDIR}/DistributedIR DEPENDS CommonHeaders)
if(GEESIBLING_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

add_library(DistributedIR block.cc dividegraph.cpp)
target_link_libraries(DistributedIR DistributedIRHeaders)
