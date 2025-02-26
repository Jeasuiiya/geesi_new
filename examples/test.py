import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
import optax

# 检查可用设备
print("Available devices:", jax.devices())

# 初始化层参数
def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b

# 初始化模型和数据
def init(key, layer_sizes, batch_size):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))
    
    keys = jax.random.split(key, 2)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))
    
    return params, (inputs, targets)

# 定义网络结构和批次大小
layer_sizes = [64, 32, 16, 8]  # 更小的网络
batch_size = 16

# 初始化参数和数据
params, batch = init(jax.random.PRNGKey(0), layer_sizes, batch_size)

# 创建设备网格
devices = mesh_utils.create_device_mesh((8,))  # 使用 8 个设备
mesh = Mesh(devices, ('feats',))

# 定义并行化的矩阵乘法函数，确保在 mesh 定义之后
@partial(shard_map, mesh=mesh,
         in_specs=(P(None, 'feats'), P('feats', None), P('feats')),
         out_specs=P(None, 'feats'))
def gemm_tp(inputs, W, b):
    block_result = jnp.dot(inputs, W)
    return jax.lax.psum_scatter(block_result, 'feats',
                                scatter_dimension=1, tiled=True) + b

# 定义张量并行的预测函数
def predict_tp(params, inputs):
    for W, b in params:
        outputs = gemm_tp(inputs, W, b)
        inputs = jax.nn.relu(outputs)
    return outputs

# 定义张量并行的损失函数
def loss_tp(params, batch):
    inputs, targets = batch
    predictions = predict_tp(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# 分片参数和数据
params_sharded = jax.device_put(params, NamedSharding(mesh, P('feats')))
batch_sharded = jax.device_put(batch, NamedSharding(mesh, P(None, 'feats')))

# 定义优化器
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params_sharded)

# 定义训练步骤
@partial(jax.pmap, axis_name='feats')
def train_step(params, batch, opt_state):
    loss, grads = jax.value_and_grad(loss_tp)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 打印 gemm_tp 的 JAXPR
jaxpr_gemm_tp = jax.make_jaxpr(gemm_tp)(batch_sharded, params_sharded[0][0], params_sharded[0][1])
print("JAXPR for gemm_tp:")
print(jaxpr_gemm_tp)

# 训练循环
num_epochs = 2  # 为了简化，仅训练 2 个 epoch

for epoch in range(num_epochs):
    params_sharded, opt_state, loss_val = train_step(params_sharded, batch_sharded, opt_state)
    
    # 计算所有设备的损失的平均值
    loss_mean = jax.device_get(jnp.mean(loss_val))
    print(f"Epoch {epoch}, Loss: {loss_mean}")
    
    # 在第一个 epoch 打印 train_step 的 JAXPR
    if epoch == 0:
        jaxpr_train_step = jax.make_jaxpr(train_step)(params_sharded, batch_sharded, opt_state)
        print("JAXPR for train_step:")
        print(jaxpr_train_step)
