# require CPM
include(CPM)

# fetch_result
function(fetch_result)
  if(NOT TARGET Result::Result)
    CPMAddPackage(
      NAME result
      GITHUB_REPOSITORY bitwizeshift/result
      GIT_TAG v1.0.0
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_result)
