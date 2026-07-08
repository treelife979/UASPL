import random

import numpy as np
import torch

MAIN_SEED = 2025  # 主种子（固定，保证全程可复现）
EXP_NUM = 5  # 实验次数

# 固定主种子，生成独立实验种子
random.seed(MAIN_SEED)
np.random.seed(MAIN_SEED)
torch.manual_seed(MAIN_SEED)

# 生成 5 个独立、不连续、高质量种子
experiment_seeds = [random.randint(10000, 99999) for _ in range(EXP_NUM)]
print(f"生成{EXP_NUM}个独立种子：{experiment_seeds}\n")

# 生成5个独立种子：[83105, 20839, 94652, 72600, 32712]