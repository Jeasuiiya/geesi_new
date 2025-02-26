include("/home/ai/ljj/tp/cmake/third_party/CPM.cmake")
CPMAddPackage("NAME;fmt;GITHUB_REPOSITORY;fmtlib/fmt;GIT_TAG;9.1.0;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;ON")
set(fmt_FOUND TRUE)