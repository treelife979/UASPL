import os
from torchvision import datasets, transforms


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

SUPPORTED_DATASETS = ("cifar10", "FashionMNIST", "MNIST", "SVHN")


def get_dataset(data_name, seed=None, root=DATA_DIR):
    """Load the four public datasets used in the paper experiments."""
    if data_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {data_name}. "
            f"Supported public datasets are: {', '.join(SUPPORTED_DATASETS)}"
        )

    normalize_cifar = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    train_transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_cifar,
    ])
    test_transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        normalize_cifar,
    ])

    train_transform_fashion = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform_fashion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_transform_svhn = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992]),
    ])
    test_transform_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992]),
    ])

    if data_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform_cifar
        )
        test_dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform_cifar
        )
    elif data_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=train_transform_fashion
        )
        test_dataset = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=test_transform_fashion
        )
    elif data_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=root, train=True, download=True, transform=transform_mnist
        )
        test_dataset = datasets.MNIST(
            root=root, train=False, download=True, transform=transform_mnist
        )
    elif data_name == "SVHN":
        train_dataset = datasets.SVHN(
            root=root, split="train", download=True, transform=train_transform_svhn
        )
        test_dataset = datasets.SVHN(
            root=root, split="test", download=True, transform=test_transform_svhn
        )

    return train_dataset, test_dataset
