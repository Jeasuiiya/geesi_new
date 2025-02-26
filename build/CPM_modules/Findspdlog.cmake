include("/home/ai/ljj/tp/cmake/third_party/CPM.cmake")
CPMAddPackage("NAME;spdlog;GITHUB_REPOSITORY;gabime/spdlog;GIT_TAG;v1.11.0;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;ON")
set(spdlog_FOUND TRUE)