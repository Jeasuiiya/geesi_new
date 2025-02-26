## Geesibling-Jax

#### 并行方式

##### 流水线并行

```python
# set method	
method=PipeshardParallel(
        policy="sgp",
        num_microbatch=4,
        layer_method="auto",
        if_ray=True)
@parallelize(parallel_method=method)
def update(i, opt_state, batch):
   # start train
```

##### 模型并行

```python
# set method	
method=ShardParallel(
    policy="sgp",
    devices=device_config(
    {
    "gpu:0": {
    "type": DeviceType.gpu,
    "memory": 3 * 1024 * 1024* 1024 ,
    "free_memory":3 * 1024 * 1024* 1024 ,
    "execute_time": 0,
    },
    }
    ) )
@parallelize(parallel_method=method)
def update(i, opt_state, batch):
   # start train
```

#### 执行过程

##### 编译

```sh
pip install -r requirements-dev.txt
./scripts/configure
cd build && ninja
```

##### 导入环境

```sh
export PYTHONPATH=<PROJECT_ROOT>/python:<PROJECT_ROOT>/build/python
```

##### 执行

```python 
# 主节点
ray start --head
python mnist.py
#从节点
ray start --address="ip:port"
```

#### 支持模型

1. vgg16
2. bert-110M
3. gpt2-117M
4. t5 (small,base)

$$

$$

