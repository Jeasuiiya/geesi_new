load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def geesibling_workspace0():
    git_repository(
        name = "fmt",
        tag = "9.1.0",
        remote = "https://github.com/fmtlib/fmt",
        patch_cmds = [
            "mv support/bazel/.bazelrc .bazelrc",
            "mv support/bazel/.bazelversion .bazelversion",
            "mv support/bazel/BUILD.bazel BUILD.bazel",
            "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
        ],
        # Windows-related patch commands are only needed in the case MSYS2 is not installed.
        # More details about the installation process of MSYS2 on Windows systems can be found here:
        # https://docs.bazel.build/versions/main/install-windows.html#installing-compilers-and-language-runtimes
        # Even if MSYS2 is installed the Windows related patch commands can still be used.
        patch_cmds_win = [
            "Move-Item -Path support/bazel/.bazelrc -Destination .bazelrc",
            "Move-Item -Path support/bazel/.bazelversion -Destination .bazelversion",
            "Move-Item -Path support/bazel/BUILD.bazel -Destination BUILD.bazel",
            "Move-Item -Path support/bazel/WORKSPACE.bazel -Destination WORKSPACE.bazel",
        ],
    )

    git_repository(
        name = "range-v3",
        commit = "d04bfc", # Commits on Mar 3, 2023
        remote = "https://github.com/ericniebler/range-v3"
    )

    new_git_repository(
        name = "result",
        tag = "v1.0.0",
        remote = "https://github.com/bitwizeshift/result",
        build_file = "//:bazel/result.BUILD"
    )

    new_git_repository(
        name = "spdlog",
        tag = "v1.12.0",
        remote = "https://github.com/gabime/spdlog",
        build_file = "//:bazel/spdlog.BUILD"
    )

    http_archive(
        name = "global_pybind11_bazel",
        strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.zip"],
    )

