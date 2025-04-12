#: <<'COMMENT'
# 预期的节点数量（包括主节点）
EXPECTED_NODES=2

# 等待 Ray 服务启动并等待从节点加入集群
MAX_ATTEMPTS=15
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    NODE_COUNT=$(python - <<EOF
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(('127.0.0.1', 6379))
    s.close()
    import ray
    ray.init(address='auto')
    nodes = ray.nodes()
    print(len(nodes))
    ray.shutdown()
except Exception:
    print(0)
EOF
)
    if [ "$NODE_COUNT" -eq "$EXPECTED_NODES" ]; then
        echo "所有节点已加入集群，开始运行 Python 脚本。"
        echo
        break
    else
        echo "等待从节点加入集群，当前节点数量: $NODE_COUNT，尝试次数: $ATTEMPT"
        sleep 5
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "达到最大尝试次数，仍未等到所有节点加入集群，退出。"
    ray stop
    exit 1
fi
#COMMENT

# 运行 Python 脚本
python run_clm_flax4.py \
       --output_dir="./norwegian-gpt2" \
       --model_type="gpt2" \
       --config_name="./norwegian-gpt2" \
       --tokenizer_name="./norwegian-gpt2" \
       --dataset_name="oscar" \
       --do_train \
       --do_eval \
       --block_size="16" \
       --per_device_train_batch_size="16" \
       --per_device_eval_batch_size="8" \
       --learning_rate="5e-3" \
       --warmup_steps="1000" \
       --adam_beta1="0.9" \
       --adam_beta2="0.98" \
       --weight_decay="0.01" \
       --overwrite_output_dir \
       --num_train_epochs="10" \
       --logging_steps="500" \
       --save_steps="2500" \
       --eval_steps="2500"

