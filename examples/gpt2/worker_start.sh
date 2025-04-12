#!/bin/bash

# 停止 Ray 服务，忽略可能的警告信息
ray stop 2>/dev/null

# 导入环境变量
export PYTHONPATH=/home/ai/ljj/geesi_new/python:/home/ai/ljj/geesi_new/build/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 开启监测权限
sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'

# 请根据实际情况修改主节点的 IP 和端口
MASTER_IP="172.20.21.5"
MASTER_PORT="6379"

# 启动 Ray 从节点服务
ray start --address="${MASTER_IP}:${MASTER_PORT}"