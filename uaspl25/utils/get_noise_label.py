import numpy as np

from utils.global_seed import set_global_seed


def add_noise(y_train, corruption_type, corruption_ratio, num_classes, seed=None):
    set_global_seed(seed)

    if not 0 <= corruption_ratio <= 1:
        raise ValueError(f"corruption_ratio must be in [0, 1], got {corruption_ratio}")
    if corruption_type is None or corruption_ratio == 0:
        return y_train

    corruption_list = {
        "uniform": lambda r, n: np.full((n, n), 1 / n) * r + np.eye(n) * (1 - r),
        "asn": asn_corruption,
    }
    if corruption_type not in corruption_list:
        raise ValueError(f"Unsupported corruption_type: {corruption_type}")

    if corruption_type == "uniform":
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
    else:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes, seed)

    y_noisy = y_train.copy()
    for idx, original_label in enumerate(y_noisy):
        y_noisy[idx] = np.random.choice(num_classes, p=corruption_matrix[original_label])
    return y_noisy


def asn_corruption(corruption_ratio, num_classes, _seed=None):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    for i in range(num_classes):
        corruption_matrix[i][(i + 1) % num_classes] = corruption_ratio
    return corruption_matrix