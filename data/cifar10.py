import torch

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision import datasets
from torchvision import transforms


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
    num_workers=4,
    seed=42,
    root="datasets",
):

    # =================================================
    # VALIDATION
    # =================================================

    if validation_size != 5000:

        raise ValueError(
            "For stratified CIFAR-10 split, "
            "validation_size must be 5000."
        )

    # =================================================
    # TRANSFORMS
    # =================================================

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

    # =================================================
    # DATASETS
    # =================================================

    train_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=test_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )

    # =================================================
    # STRATIFIED TRAIN / VALIDATION SPLIT
    # =================================================

    generator = torch.Generator().manual_seed(
        seed
    )

    train_indices = []
    validation_indices = []

    targets = torch.tensor(
        train_dataset_full.targets
    )

    num_classes = 10

    validation_per_class = (
        validation_size
        // num_classes
    )

    train_per_class = (
        5000
        - validation_per_class
    )

    for class_index in range(
        num_classes
    ):

        class_indices = torch.where(
            targets == class_index
        )[0]

        # Shuffle indices of current class

        permutation = torch.randperm(
            len(class_indices),
            generator=generator,
        )

        class_indices = class_indices[
            permutation
        ]

        # Validation samples

        class_validation_indices = (
            class_indices[
                :validation_per_class
            ]
        )

        # Training samples

        class_train_indices = (
            class_indices[
                validation_per_class:
            ]
        )

        validation_indices.extend(
            class_validation_indices.tolist()
        )

        train_indices.extend(
            class_train_indices.tolist()
        )

    # =================================================
    # SHUFFLE FINAL INDICES
    # =================================================

    train_indices = torch.tensor(
        train_indices
    )

    validation_indices = torch.tensor(
        validation_indices
    )

    train_permutation = torch.randperm(
        len(train_indices),
        generator=generator,
    )

    validation_permutation = torch.randperm(
        len(validation_indices),
        generator=generator,
    )

    train_indices = train_indices[
        train_permutation
    ]

    validation_indices = validation_indices[
        validation_permutation
    ]

    # =================================================
    # CREATE SUBSETS
    # =================================================

    train_dataset = Subset(
        train_dataset_full,
        train_indices.tolist(),
    )

    validation_dataset = Subset(
        validation_dataset_full,
        validation_indices.tolist(),
    )

    # =================================================
    # VERIFY SPLIT
    # =================================================

    train_indices_set = set(
        train_indices.tolist()
    )

    validation_indices_set = set(
        validation_indices.tolist()
    )

    overlap = (
        train_indices_set
        & validation_indices_set
    )

    assert len(overlap) == 0, (
        "Train and validation datasets overlap."
    )

    assert len(train_indices) == 45000, (
        "Train dataset must contain 45000 samples."
    )

    assert len(validation_indices) == 5000, (
        "Validation dataset must contain 5000 samples."
    )

    # =================================================
    # CREATE LOADERS
    # =================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    # check_cifar10_split(train_dataset,validation_dataset,test_dataset)
    # check_dataset_overlap(train_dataset,validation_dataset,test_dataset)

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


def check_cifar10_split(
    train_dataset,
    validation_dataset,
    test_dataset,
):
    from collections import Counter
    # ================================================
    # TRAIN
    # ================================================

    train_indices = train_dataset.indices

    train_targets = [
        train_dataset.dataset.targets[index]
        for index in train_indices
    ]

    # ================================================
    # VALIDATION
    # ================================================

    validation_indices = (
        validation_dataset.indices
    )

    validation_targets = [
        validation_dataset.dataset.targets[index]
        for index in validation_indices
    ]

    # ================================================
    # TEST
    # ================================================

    test_targets = test_dataset.targets

    # ================================================
    # PRINT CLASS DISTRIBUTION
    # ================================================

    print(
        "Train:"
    )

    print(
        Counter(train_targets)
    )

    print()

    print(
        "Validation:"
    )

    print(
        Counter(validation_targets)
    )

    print()

    print(
        "Test:"
    )

    print(
        Counter(test_targets)
    )

def check_dataset_overlap(
    train_dataset,
    validation_dataset,
    test_dataset,
):

    train_indices = set(
        train_dataset.indices
    )

    validation_indices = set(
        validation_dataset.indices
    )

    # ================================================
    # TRAIN / VALIDATION
    # ================================================

    train_validation_overlap = (
        train_indices
        & validation_indices
    )

    print(
        "Train / Validation overlap:",
        len(train_validation_overlap),
    )

    # ================================================
    # SIZES
    # ================================================

    print(
        "Train size:",
        len(train_indices),
    )

    print(
        "Validation size:",
        len(validation_indices),
    )

    print(
        "Test size:",
        len(test_dataset),
    )

    # ================================================
    # ASSERTIONS
    # ================================================

    assert len(train_validation_overlap) == 0, (
        "Train and validation datasets overlap!"
    )

    assert len(train_indices) == 45000

    assert len(validation_indices) == 5000

    assert len(test_dataset) == 10000

    print(
        "No overlap between train and validation."
    )