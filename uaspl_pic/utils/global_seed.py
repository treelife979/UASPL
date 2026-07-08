import os
import random
import numpy as np
import torch

# 设置全局种子
def set_global_seed(seed):
    random.seed(seed) # 1. 固定 Python 自带随机数生成器的种子
    np.random.seed(seed) # 2. 固定 NumPy 随机数生成器的种子
    torch.manual_seed(seed) # 3. 固定 PyTorch CPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)  # 4. 固定所有 GPU 上的随机数种子（多卡/单卡都生效）
    torch.backends.cudnn.deterministic = True # 5. 让 CuDNN 使用确定性算法，禁用优化带来的随机性
    torch.backends.cudnn.benchmark = False # 6. 关闭 CuDNN 自动寻找最优卷积算法的功能

# 2. 固定 DataLoader 多进程随机种子（会影响shffle=True时数据的可复现性）
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # % 2**32：取模运算（2**32 是 32 位整数的最大值），取模后把值限制在 32 位范围内
    np.random.seed(worker_seed) # 设置 NumPy 随机数生成器的种子
    random.seed(worker_seed) # 设置 Python 内置随机数生成器的种子