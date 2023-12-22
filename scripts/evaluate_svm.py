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

from informed_classification.analysis import evaluate_svm_model
from informed_classification.common_utilities import get_config, load_data

config, _ = get_config()
print(config)
quit()
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
evaluate_svm_model(svm_model, X_train, y_train, "Train Set")
evaluate_svm_model(svm_model, X_test, y_test, "Test Set")
evaluate_svm_model(svm_model, X_val, y_val, "Validation Set")
