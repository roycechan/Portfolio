# Net architecture
net_size = {
    0.4: [24,48,192,1024],
    0.5: [24, 48, 96, 192, 1024],
    1: [24, 116, 232, 464, 1024],
    1.5: [24, 176, 352, 704, 1024],
    2: [24, 224, 976, 2048],
    4: [24, 488, 976, 2048, 4096]
}
net_blocks = [3, 3]
# Stage
conv1_kernel_size = 2
conv1_stride = 1
conv5_kernel_size = 1
conv5_stride = 1
global_pool_kernel_size = 7

# Data
num_classes = 2300
# Training parameters
lr1 = 1e-3
weight_decay = 1e-4
lr2 = lr1 / 10
batch_size = 256
max_epoch = 20
# Model
net_size_chosen = 4

