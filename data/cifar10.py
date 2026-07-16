import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def get_cifar10_loaders(
    batch_size=128,
    validation_size=5000,
    num_workers=8,
    seed=42,
    root="datasets",
):

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    full_train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=test_transform,
    )

    train_size = len(full_train_dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, _ = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    _, validation_dataset = random_split(
        validation_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )