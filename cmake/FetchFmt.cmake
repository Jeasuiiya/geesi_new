# require CPM
include(CPM)

# fetch_result
function(fetch_fmt)
  if(NOT TARGET fmt::fmt)
    CPMAddPackage(
      NAME fmt
      GITHUB_REPOSITORY fmtlib/fmt
      GIT_TAG 9.1.0
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_fmt)
