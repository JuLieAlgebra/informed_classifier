import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.svm import SVC

from informed_classification.common_utilities import get_config, load_data


def evaluate_svm_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Metrics for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # ROC Curve
    y_score = model.decision_function(X)
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic for {dataset_name}")
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()


config = get_config()

# Load datasets
train_data = load_data(f"data/{config['experiment_name']}/train")
test_data = load_data(f"data/{config['experiment_name']}/test")
validation_data = load_data(f"data/{config['experiment_name']}/validation")

# Split features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
X_val, y_val = validation_data[:, :-1], validation_data[:, -1]

# SVM with Gaussian RBF Kernel
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)

# Evaluating on Test and Validation Sets
evaluate_model(svm_model, X_train, y_train, "Train Set")
evaluate_model(svm_model, X_test, y_test, "Test Set")
evaluate_model(svm_model, X_val, y_val, "Validation Set")
