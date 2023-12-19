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

from informed_classification.common_utilities import get_config, load_data

#### Config
config, _ = get_config()

# Constants
EXPERIMENT_NAME = config["experiment_name"]
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_SIZES = [100, 1000, 10000]
KFOLDS = 5  # Number of folds in KFold cross-validation


############################################################


def compute_nn_metrics(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    y_pred, y_true, y_scores = [], [], []

    with torch.no_grad():
        for X_tensor, y_tensor in dataloader:
            outputs = model(X_tensor)
            y_scores.extend(outputs.squeeze().numpy())
            y_pred.extend(outputs.squeeze().numpy() >= 0.5)
            y_true.extend(y_tensor.numpy())

    # Convert to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }
    return metrics


def plot_nn_metrics(metrics, dataset_name):
    # ROC Curve
    # fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_scores"])
    # plt.figure()
    # plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {metrics['roc_auc']:.2f})")
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title(f"Receiver operating characteristic for {dataset_name}")
    # plt.legend(loc="lower right")
    # plt.show()

    # Confusion Matrix
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()


# def compute_nn_metrics(model, dataloader, dataset_name):
#     model.eval()  # Set the model to evaluation mode
#     y_pred = []
#     y_true = []
#     y_scores = []

#     with torch.no_grad():
#         for X_tensor, y_tensor in dataloader:
#             outputs = model(X_tensor)
#             y_scores.extend(outputs.squeeze().numpy())
#             y_pred.extend(outputs.squeeze().numpy() >= 0.5)
#             y_true.extend(y_tensor.numpy())

#     # Convert to numpy arrays for metric calculation
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     y_scores = np.array(y_scores)

#     # Calculate metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_scores)

#     print(f"Metrics for {dataset_name}:")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1 Score: {f1:.2f}")
#     print(f"ROC AUC: {roc_auc:.2f}")

#     # ROC Curve
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     plt.figure()
#     plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
#     plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"Receiver operating characteristic for {dataset_name}")
#     plt.legend(loc="lower right")
#     plt.show()

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title(f"Confusion Matrix for {dataset_name}")
#     plt.ylabel("Actual label")
#     plt.xlabel("Predicted label")
#     plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(nn_model, dataloader, epochs: int):
    # Loss and optimizer
    criterion = nn.BCELoss()
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

    for sample_size in sample_sizes:
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_performance = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
            print(f"Training on {sample_size} samples - Fold {fold+1}/{k_folds}")

            # Subsample the training and validation sets
            train_subsampler = Subset(full_train_dataset, train_ids[:sample_size])
            val_subsampler = Subset(full_train_dataset, val_ids)

            train_loader = DataLoader(
                train_subsampler, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize model
            nn_model = ShallowNN(input_size)
            train(nn_model, train_loader, EPOCHS)

            # Evaluate on validation set
            performance_metrics = compute_nn_metrics(
                nn_model, val_loader
            )  # , f"Fold {fold+1} Validation")
            fold_performance.append(performance_metrics)
            plot_nn_metrics(performance_metrics, "Training Set")

        # # Calculate average performance metrics across folds
        # avg_performance = np.mean(fold_performance, axis=0)
        # print(f"Average performance for sample size {sample_size}: {avg_performance}")


kfold_training_sample_sweep(SAMPLE_SIZES, KFOLDS)

# # Load training data
# train_data_dir = f"data/{EXPERIMENT_NAME}/train"
# train_dataset = CustomDataset(train_data_dir)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Initialize the Neural Network
# input_size = train_dataset[0][0].shape[0]  # Get the size of the input features
# nn_model = ShallowNN(input_size)

# train(nn_model, train_loader, epochs=EPOCHS)
# compute_nn_metrics(nn_model, train_loader, "Training Set")

# # Load test data
# test_data_dir = f"data/{EXPERIMENT_NAME}/test"
# test_dataset = CustomDataset(test_data_dir)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# compute_nn_metrics(nn_model, test_loader, "Test Set")

# # Load validation data
# validation_data_dir = f"data/{EXPERIMENT_NAME}/validation"
# validation_dataset = CustomDataset(validation_data_dir)
# validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
# compute_nn_metrics(nn_model, validation_loader, "Validation Set")
