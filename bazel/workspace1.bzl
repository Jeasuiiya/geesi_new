load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@global_pybind11_bazel//:python_configure.bzl", "python_configure")
def geesibling_workspace1():
    python_configure(name = "global_config_python", python_version = "3")
    http_archive(
        name = "global_pybind11",
        build_file = "@global_pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-2.10.3",
        urls = ["https://github.com/pybind/pybind11/archive/v2.10.3.tar.gz"],
        repo_mapping = {
            "@local_config_python" : "@global_config_python"
        }
    )
    git_repository(
        name = "org_tensorflow",
        tag = "v2.11.0",
        remote = "https://github.com/tensorflow/tensorflow",
        patches = ["//:bazel/tf.patch"],
        patch_cmds = [
            "echo \"exports_files(['tools/tf_env_collect.sh', '.bazelrc'])\" >> BUILD",
            "echo \"exports_files(['find_cuda_config.py', 'find_rocm_config.py'])\" >> third_party/gpus/BUILD",
        ]
    )