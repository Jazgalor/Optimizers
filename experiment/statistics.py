import json


class ExperimentStatistics:

    def __init__(
        self,
        model_name,
        optimizer_name,
        epochs,
        batch_size,
        seed,
    ):

        # ============================================
        # Experiment information
        # ============================================

        self.model_name = model_name
        self.optimizer_name = optimizer_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # ============================================
        # Per epoch statistics
        # ============================================

        self.train_loss_history = []
        self.train_accuracy_history = []

        self.validation_loss_history = []
        self.validation_accuracy_history = []

        self.epoch_time_history = []

        # ============================================
        # Overall statistics
        # ============================================

        self.total_training_time = 0.0

        self.test_loss = 0.0
        self.test_accuracy = 0.0

    def to_dict(self):

        return {

            "model_name": self.model_name,
            "optimizer_name": self.optimizer_name,

            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,

            "train_loss_history": self.train_loss_history,
            "train_accuracy_history": self.train_accuracy_history,

            "validation_loss_history": self.validation_loss_history,
            "validation_accuracy_history": self.validation_accuracy_history,

            "epoch_time_history": self.epoch_time_history,

            "total_training_time": self.total_training_time,

            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
        }

    def save(
        self,
        path,
    ):

        with open(
            path,
            "w",
        ) as file:

            json.dump(
                self.to_dict(),
                file,
                indent=4,
            )