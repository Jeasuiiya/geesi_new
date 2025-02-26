load("@global_pybind11_bazel//:build_defs.bzl", "pybind_extension")

def py_extension(name, defines = [], **kwargs):
    pybind_extension(
        name = name,
        defines = ["PYBIND11_CURRENT_MODULE_NAME=%s" % name] + defines,
        **kwargs
    )
    native.py_library(
        name = name,
        data = [name + ".so"]
    )
