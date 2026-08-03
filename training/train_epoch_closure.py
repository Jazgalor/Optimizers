import time

import torch


def train_epoch_closure(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    start_time = time.perf_counter()

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        closure_data = {
            "loss": None,
            "outputs": None,
        }

        
        def closure(mode="train"):

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            if mode == "train":

                loss = criterion(outputs, labels)

            elif mode == "gnb":

                probs = torch.softmax(outputs, dim=1)

                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1
                ).squeeze(1)

                loss = criterion(
                    outputs,
                    sampled_labels
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss.backward()

            closure_data["loss"] = loss.detach()
            closure_data["outputs"] = outputs.detach()
            closure_data["batch_size"] = images.size(0)

            return loss

        optimizer.step(closure)

        running_loss += (
            closure_data["loss"].item()
            * labels.size(0)
        )

        predictions = (
            closure_data["outputs"]
            .argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_time = (
        time.perf_counter()
        - start_time
    )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_time,
    )