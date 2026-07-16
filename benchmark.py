import os
import time

import torch
import torch.nn as nn

from models.resnet import ResNet20

from opt_torch.sgd  import SGDTorch

from data.cifar10 import get_cifar10_loaders

from experiment.statistics import ExperimentStatistics

from training.train_epoch import train_epoch
from training.train_epoch_closure import train_epoch_closure
from training.evaluate import evaluate


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42

EPOCHS = 200

BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    # =================================================
    # RANDOM SEED
    # =================================================

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ================================================
    # MODEL
    # ================================================

    model = ResNet20().to(DEVICE)

    # ================================================
    # OPTIMIZER
    # ================================================

    optimizer = SGDTorch(
        model.parameters()
    )

    # ================================================
    # LOSS FUNCTION
    # ================================================

    criterion = nn.CrossEntropyLoss()

    # ================================================
    # DATA
    # ================================================

    train_loader, validation_loader, test_loader = (
        get_cifar10_loaders(
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    # ================================================
    # STATISTICS
    # ================================================

    statistics = ExperimentStatistics(
        model_name=model.__class__.__name__,
        optimizer_name=optimizer.__class__.__name__,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # ================================================
    # CLOSURE
    # ================================================

    closure = None

    # ================================================
    # TRAINING
    # ================================================

    training_start = time.perf_counter()

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        if closure is None:

            (
                train_loss,
                train_accuracy,
                epoch_time,
            ) = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
            )

        else:

            (
                train_loss,
                train_accuracy,
                epoch_time,
            ) = train_epoch_closure(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                closure=closure,
                device=DEVICE,
            )

        (
            validation_loss,
            validation_accuracy,
        ) = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=DEVICE,
        )

        statistics.train_loss_history.append(
            train_loss
        )

        statistics.train_accuracy_history.append(
            train_accuracy
        )

        statistics.validation_loss_history.append(
            validation_loss
        )

        statistics.validation_accuracy_history.append(
            validation_accuracy
        )

        statistics.epoch_time_history.append(
            epoch_time
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    statistics.total_training_time = (
        time.perf_counter()
        - training_start
    )

    # ================================================
    # TEST
    # ================================================

    (
        test_loss,
        test_accuracy,
    ) = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
    )

    statistics.test_loss = test_loss
    statistics.test_accuracy = test_accuracy

    # ================================================
    # SAVE
    # ================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )

    experiment_name = (
        f"{model.__class__.__name__}"
        f"_{optimizer.__class__.__name__}"
    )

    torch.save(
        model.state_dict(),
        f"results/{experiment_name}.pth",
    )

    statistics.save(
        f"results/{experiment_name}.json",
    )

    # ================================================
    # SUMMARY
    # ================================================

    print("\nTraining finished.")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":

    main()