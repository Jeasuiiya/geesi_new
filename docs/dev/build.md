# Build

## Tensorflow

### Debug
Debug工具`tf-run-graph`, 执行tf1计算图
```bash
# 依赖
pip install -r requirements-dev.txt
./scripts/build-tf --config-only
bazel build tf-run-graph --config=dbg
bazel build rpc_server --config=dbg
```

运行`tf-run-graph`
```bash
# 依赖
pip install tensorflow networkx
export PYTHONPATH=$PWD/python
./bazel-bin/rpc_server localhost:9001
TF_PLACEMENT_RPC_ADDRESS=localhost:9001 ./bazel-bin/tf-run-graph --graph=/path/to/graph --train_op=<trainop>
```

**传递给Pass的参数**

环境变量

`TF_PLACEMENT_POLICY`: 可选值：`fddps`, `sgp`



**模型文件**

* [resnet50](https://gist.githubusercontent.com/Yiklek/cc66295cef7361c6a701c9408f1e2661/raw/c7e1bc36f178d57480ae701bc4f7a11cc5b1a530/resnet50.pbtxt)

### Release

打包Tensorflow到dist
```bash
# 依赖
pip install -r requirements-dev.txt
./scripts/build-tf
```

## Jax

### 编译

需要至少`g++-9`以上

其他依赖

* cmake
* ninja

```bash
pip install -r requirements-dev.txt
./scripts/configure
cd build && ninja
```

### 运行环境

**依赖**
jax>=0.4
jaxlib>=0.4

dcu 只能使用0.3版本，但0.3版本未测试

**环境变量**

暂未提供`setup.py`安装方式，需要将路径`python`注册为`python`的搜索路径

```bash
export PYTHONPATH=<PROJECT_ROOT>/python:<PROJECT_ROOT>/build/python
```

### 并行化

对需要并行化的函数使用`parallelize`进行装饰，可参考[example/mnist.py](../../examples/mnist.py)
