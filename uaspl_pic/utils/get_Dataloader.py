import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.global_seed import seed_worker, set_global_seed


class LabelModifyDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = np.array([sample[1] for sample in base_dataset])

    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        return data, self.labels[idx]

    def __len__(self):
        return len(self.base_dataset)


def get_Dataloader(
    train_dataset,
    test_dataset,
    num_classes,
    batch_size,
    num_workers,
    imbalanced_factor=None,
    corruption_type=None,
    corruption_ratio=0.0,
    seed=1,
):

    return build_dataloader_no_meta(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        imbalanced_factor=imbalanced_factor,
        corruption_type=corruption_type,
        corruption_ratio=corruption_ratio,
        seed=seed,
    )


def build_dataloader_no_meta(
    train_dataset,
    test_dataset,
    num_classes,
    batch_size=64,
    num_workers=2,
    imbalanced_factor=None,
    corruption_type=None,
    corruption_ratio=0.0,
    seed=1,
):
    assert 0 <= corruption_ratio <= 1, "corruption_ratio must be in [0, 1]"
    set_global_seed(seed)

    all_labels = get_all_labels(train_dataset)
    index_to_train = []

    if imbalanced_factor is not None:
        imbalanced_num_list = []
        sample_num = int(len(train_dataset) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (
                imbalanced_factor ** (class_index / (num_classes - 1))
            )
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        class_indices = [
            index for index, label in enumerate(all_labels) if int(label) == class_index
        ]
        if imbalanced_num_list is not None:
            class_indices = class_indices[: imbalanced_num_list[class_index]]
        index_to_train.extend(class_indices)

    base_train_dataset = Subset(train_dataset, index_to_train)

    if corruption_type is not None and corruption_ratio > 0:
        final_train_dataset = LabelModifyDataset(base_train_dataset)
        corruption_list = {
            "uniform": lambda r, n: np.full((n, n), 1 / n) * r + np.eye(n) * (1 - r),
            "flip1": flip1_corruption,
            "flip2": flip2_corruption,
        }
        if corruption_type not in corruption_list:
            raise ValueError(f"Supported corruption types: {list(corruption_list.keys())}")
        if corruption_type == "uniform":
            corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        else:
            corruption_matrix = corruption_list[corruption_type](
                corruption_ratio, num_classes, seed
            )
        for idx in range(len(final_train_dataset)):
            original_label = int(final_train_dataset.labels[idx])
            final_train_dataset.labels[idx] = np.random.choice(
                num_classes, p=corruption_matrix[original_label]
            )
    else:
        final_train_dataset = base_train_dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled_indices = torch.randperm(len(final_train_dataset), generator=generator).tolist()
    shuffled_train_dataset = Subset(final_train_dataset, shuffled_indices)

    train_dataloader = DataLoader(
        shuffled_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    return shuffled_train_dataset, train_dataloader, test_dataloader


def flip1_corruption(corruption_ratio, num_classes, seed):
    rng = np.random.RandomState(seed)
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][rng.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes, seed):
    rng = np.random.RandomState(seed)
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][rng.choice(row_indices[row_indices != i], 2, replace=False)] = (
            corruption_ratio / 2
        )
    return corruption_matrix


def get_all_labels(dataset):
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(label)
    return labels
