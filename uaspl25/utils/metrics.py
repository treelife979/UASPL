import numpy as np
import torch


def scalar_to_float(x):
    """
    将不同来源的标量统一转换为 Python float。

    兼容：
    1. Python int / float
    2. numpy 标量，例如 np.float32 / np.float64
    3. torch 标量 tensor
    4. 只包含一个元素的 list / tuple / ndarray / tensor
    """
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())

    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0].item())

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Cannot convert an empty list/tuple to float.")
        return scalar_to_float(x[0])

    return float(x)


def clean_metric_list(values):
    """将指标列表转换为纯 Python float，并过滤 NaN / Inf。"""
    cleaned = []
    for v in values:
        fv = scalar_to_float(v)
        if np.isfinite(fv):
            cleaned.append(fv)
    return cleaned


def mean_and_std(values):
    """
    稳健计算均值和样本标准差。

    注意：statistics.stdev 至少需要两个样本；
    当有效样本数不足 2 时，标准差返回 0.0。
    """
    values = clean_metric_list(values)
    if len(values) == 0:
        return 0.0, 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_value, std_value
