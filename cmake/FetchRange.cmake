# require CPM
include(CPM)

# fetch_result
function(fetch_range)
  if(NOT TARGET range-v3)
    CPMAddPackage(
      NAME range
      # Commits on Mar 14, 2023
      URL "https://github.com/ericniebler/range-v3/archive/541b06320b89c16787cc6f785f749f8847cf2cd1.zip"
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_range)
