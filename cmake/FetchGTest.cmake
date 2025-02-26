# require CPM
include(CPM)

# fetch_gtest
function(fetch_gtest)
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  # cmake-lint: disable=C0103
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG release-1.12.1
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_gtest)
