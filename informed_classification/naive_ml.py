"""
Demonstration of typical ML solution when there is no information available about the system in question.

Neural network.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Torch related dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from informed_classification.analysis import compute_nn_metrics, plot_boxplots
from informed_classification.common_utilities import get_config, load_data


def plot_nn_metrics(metrics, dataset_name, show=False, filepath=""):
    # Confusion Matrix
    plt.figure()
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    if show:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches="tight")


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(nn_model, dataloader, epochs: int):
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(epochs):
        for data, labels in dataloader:
            # Forward pass
            outputs = nn_model(data)
            loss = criterion(outputs.squeeze(), labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("Training complete.")


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data, self.labels = self.load_data(data_dir)

    def load_data(self, data_dir):
        data = []
        labels = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".npy"):
                sample = np.load(os.path.join(data_dir, filename))
                data.append(sample[:-1])  # Assuming the last element is the label
                labels.append(sample[-1])
        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


class ShallowNN(nn.Module):
    def __init__(self, input_size):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def kfold_training_sample_sweep(sample_sizes, k_folds):
    train_data_dir = f"data/{EXPERIMENT_NAME}/train"
    full_train_dataset = CustomDataset(train_data_dir)
    input_size = full_train_dataset[0][0].shape[0]

    per_model_sample_sizes = [20, 40, 100, 200, 300, 500, 800, 1000, 2000]
    sample_sizes = [s * 2 for s in per_model_sample_sizes]
    k_fold_metrics = {
        size: {"train": [], "test": [], "val": []} for size in sample_sizes
    }

    for sample_size in sample_sizes:
        kfold = KFold(n_splits=k_folds, shuffle=True)
        training_dataset_section = full_train_dataset[:sample_size]
        for fold, (train_ids, val_ids) in enumerate(
            kfold.split(training_dataset_section)
        ):
            print(f"Training on {sample_size} samples - Fold {fold+1}/{k_folds}")

            # Subsample the training and validation sets
            train_subsampler = Subset(full_train_dataset, train_ids[:sample_size])
            val_subsampler = Subset(full_train_dataset, val_ids)

            train_loader = DataLoader(
                train_subsampler, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize model
            nn_model = NeuralNetwork(input_size)
            train(nn_model, train_loader, EPOCHS)

            # Evaluate on validation set
            train_metrics, training_model_outputs = compute_nn_metrics(
                nn_model, train_loader
            )
            test_metrics, testing_model_outputs = compute_nn_metrics(
                nn_model, test_loader
            )
            val_metrics, validation_model_outputs = compute_nn_metrics(
                nn_model, val_loader
            )

            plot_cm_matrix(
                training_model_outputs["y_true"],
                training_model_outputs["y_pred"],
                n_training_samples=training_fold.shape[0],
                save=True,
                dataset_name="Training Set",
                filepath=f"data/plots/NN_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_NN_train_confusionmatrix",
            )
            plot_cm_matrix(
                testing_model_outputs["y_true"],
                testing_model_outputs["y_pred"],
                n_training_samples=training_fold.shape[0],
                save=True,
                dataset_name="Testing Set",
                filepath=f"data/plots/NN_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_NN_test_confusionmatrix",
            )
            plot_cm_matrix(
                validation_model_outputs["y_true"],
                validation_model_outputs["y_pred"],
                n_training_samples=training_fold.shape[0],
                save=True,
                dataset_name="Validation Set",
                filepath=f"data/plots/NN_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_NN_val_confusionmatrix",
            )

            k_fold_metrics[sample_size]["train"].append(train_fitted_metrics)
            k_fold_metrics[sample_size]["test"].append(test_fitted_metrics)
            k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)

    return k_fold_metrics, sample_sizes


if __name__ == "__main__":
    #### Config
    config, _ = get_config()

    # Constants
    EXPERIMENT_NAME = config["experiment_name"]
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    SAMPLE_SIZES = [50, 100, 1000, 10000]
    KFOLDS = 5  # Number of folds in KFold cross-validation

    k_fold_metrics, sample_sizes = kfold_training_sample_sweep(SAMPLE_SIZES, KFOLDS)
    plot_boxplots(k_fold_metrics, sample_sizes, model="NeuralNetwork")
