import os
import ray
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.example_libraries import stax, optimizers
import numpy as np
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf

from geesibling.adapters.jax import parallelize, device_config, DeviceType
from geesibling.adapters.jax import api
from geesibling.adapters.jax.pipeline.devicecontext import init_global_cluster
#init_global_cluster(cluster="ray")
from geesibling.adapters.jax.parallel_method import ShardParallel,PipeshardParallel
# Load and preprocess CIFAR-10 data
def load_cifar10_local(data_dir, split, is_training=True, batch_size=32):
    # 确定文件名
    if split == 'train':
        filenames = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch')]

    # 循环遍历所有批次文件
    while True:
        for filename in filenames:
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='bytes')
                X = datadict[b'data']
                Y = datadict[b'labels']
                X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
                X /= 255.0
                Y = np.array(Y)

            # 打乱数据（如果在训练模式下）
            if is_training:
                indices = np.random.permutation(len(X))
                X, Y = X[indices], Y[indices]

            # 分批次返回数据
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                yield X[start:end], Y[start:end]

# Define VGG16 architecture using stax
def VGG16(num_classes):
    init_fun, convnet = stax.serial(
        stax.Conv(64, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(64, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2)),
        stax.Conv(128, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(128, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2)),
        stax.Conv(256, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(256, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(256, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2)),
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2)),
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.Conv(512, (3, 3), (1, 1), padding="SAME"), stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2)),
        stax.Flatten,
        stax.Dense(4096), stax.Relu,
        stax.Dense(4096), stax.Relu,
        stax.Dense(num_classes), stax.LogSoftmax
    )

    return init_fun, convnet

# Define the loss function
def loss(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    return -jnp.mean(jnp.sum(targets * logits, axis=1))

# Define the accuracy metric
def accuracy(params, images, targets):
    predicted_class = jnp.argmax(predict_fun(params, images), axis=1)
    true_class = jnp.argmax(targets, axis=1)
    return jnp.mean(predicted_class == true_class)

# Initialize the model
num_classes = 10  # Number of classes in CIFAR-10
batch_size = 4096
step_size = 0.001
num_batches = 5000000000 // batch_size  # Total number of batches in CIFAR-10

init_fun, predict_fun = VGG16(num_classes)
_, init_params = init_fun(random.PRNGKey(0), (batch_size, 32, 32, 3))
init_params = jax.tree_map(lambda x: x.astype(jnp.float32), init_params)

# Setup optimizer
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(init_params)

method=PipeshardParallel(
        policy="sgp",
        num_microbatch=4,
        layer_method="auto",
        if_ray=True

   )
@parallelize(parallel_method=method)
def update(opt_state, batch):
    params = get_params(opt_state)
    return opt_update(0, api.grad(loss)(params, batch), opt_state)

num_epochs = 4
num_batches_to_use = 1  # Choose the number of batches you want to use for training
num_batches_to_validate = 10
all_time = 0

for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(num_batches_to_use):
        images, labels = next(load_cifar10_local("cifar-10-batches-py", "train", is_training=True, batch_size=batch_size))
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        batch = (images, one_hot_labels)
        opt_state = update(opt_state, batch)

    print("-------------")
    val_accuracy = []
    for i in range(num_batches_to_validate):
        val_images, val_labels = next(load_cifar10_local("cifar-10-batches-py", "test", is_training=False, batch_size=batch_size))
        val_one_hot_labels = jax.nn.one_hot(val_labels, num_classes)
        val_acc = accuracy(get_params(opt_state), val_images, val_one_hot_labels)
        val_accuracy.append(val_acc)
        # Calculate the average validation accuracy
    val_accuracy = jnp.array(val_accuracy) 
    avg_val_acc = jnp.mean(val_accuracy)
    print(f"Epoch {epoch+1}, Average Validation Accuracy: {avg_val_acc:.3f}")
    dcu_time = time.time() - start_time
    print(f"Epoch {epoch+1} used {dcu_time:0.2f} sec")    
    all_time += dcu_time
print(f"One DCU: The average time each epoch used is {all_time/num_epochs:0.2f} sec")


# Test the trained model on a subset of the test set
num_batches_to_test = 10  # Choose the number of batches to test

test_accuracy = []
for i in range(num_batches_to_test):
    test_images, test_labels = next(load_cifar10_local("cifar-10-batches-py", "test", is_training=False, batch_size=batch_size))
    test_one_hot_labels = jax.nn.one_hot(test_labels, num_classes)
    test_acc = accuracy(get_params(opt_state), test_images, test_one_hot_labels)
    test_accuracy.append(test_acc)

# Calculate the average test accuracy
test_accuracy = jnp.array(test_accuracy) 
avg_test_acc = jnp.mean(test_accuracy)
print(f"Average Test Accuracy: {avg_test_acc:.3f}")




#ray.timeline(filename="/mnt/VMSTORE/cy_data/temp/timeline2.json")
