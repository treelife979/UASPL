import torch

from Net.resnet import ResNet18


def initial_model(model_name, dataset_name, num_classes, lr=0.1):

    if dataset_name in ["cifar10", "SVHN"]:
        in_channels = 3
    elif dataset_name in ["MNIST", "FashionMNIST"]:
        in_channels = 1
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if model_name != "ResNet18":
        raise ValueError(f"Unsupported model: {model_name}")

    model = ResNet18(num_classes, in_channels)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return model, optimizer
