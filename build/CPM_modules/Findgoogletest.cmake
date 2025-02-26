include("/home/ai/ljj/tp/cmake/third_party/CPM.cmake")
CPMAddPackage("NAME;googletest;GITHUB_REPOSITORY;google/googletest;GIT_TAG;release-1.12.1;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;ON")
set(googletest_FOUND TRUE)