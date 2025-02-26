# require CPM
include(CPM)

# fetch_boost
function(fetch_boost)
  CPMAddPackage(NAME boost URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.7z
                EXCLUDE_FROM_ALL ON)
endfunction(fetch_boost)
