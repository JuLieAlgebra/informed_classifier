import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from informed_classification.analysis import (
    evaluate_svm_model,
    plot_boxplots,
    plot_cm_matrix,
)
from informed_classification.common_utilities import get_config, load_data

config, _ = get_config()

# Load datasets
train_data = load_data(f"data/{config['experiment_name']}/train")
test_data = load_data(f"data/{config['experiment_name']}/test")
validation_data = load_data(f"data/{config['experiment_name']}/validation")

# Split features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
X_val, y_val = validation_data[:, :-1], validation_data[:, -1]

# #### Evaluating True Gaussian Processes
# nominal = generative_models.NominalModel(config["dim"])
# disrupted = generative_models.DisruptedModel(config["dim"])
# true_bayes = bayes_classifier.BayesClassifier(
#     [config["ratio"], 1 - config["ratio"]], [nominal, disrupted]
# )

### If I want each sample size to be 20, then I have to give kfold roughly sample*5*2
per_model_sample_sizes = [20, 40, 100, 200, 300, 500, 800, 1000, 2000]
sample_sizes = [s * 2 for s in per_model_sample_sizes]
k_folds = 5
k_fold_metrics = {size: {"train": [], "test": [], "val": []} for size in sample_sizes}

# Scaling the dataset
scaler = StandardScaler()

# Assuming X_train, X_test, and X_val are your data splits
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

for sample_size in sample_sizes:
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_performance = []

    sample_size_len = sample_size * k_folds
    for fold, (train_ids, val_ids) in enumerate(
        kfold.split(X=X_train[:sample_size_len], y=y_train[:sample_size_len])
    ):
        print(
            f"Training approx {len(train_ids)//2} samples per class - Fold {fold+1}/{k_folds}"
        )
        training_fold = X_train_scaled[train_ids]
        # n_training_samples = len(train_ids)//2
        assert training_fold.shape[0] == len(train_ids)

        print(
            "Nominal samples in train: ",
            training_fold[y_train[train_ids] == 0].shape,
            "Disrupted samples in train: ",
            training_fold[y_train[train_ids] == 1].shape,
        )
        # if training_fold[y_train[train_ids] == 0].shape[0] == 0 or training_fold[y_train[train_ids] == 1].shape[0] ==0:
        #     continue

        # SVM with Gaussian RBF Kernel
        svm_model = SVC(kernel="rbf")
        svm_model.fit(training_fold, y_train[train_ids])

        if not os.path.exists("data/plots/svm_confusion_matrices"):
            os.makedirs("data/plots/svm_confusion_matrices")

        # Evaluating on SCALED Training, Test, and Validation Sets
        train_fitted_metrics, train_y_pred = evaluate_svm_model(
            model=svm_model,
            X=training_fold,
            y=y_train[train_ids],
            dataset_name="Train Set",
        )
        test_fitted_metrics, test_y_pred = evaluate_svm_model(
            model=svm_model, X=X_test_scaled, y=y_test, dataset_name="Test Set"
        )
        val_fitted_metrics, val_y_pred = evaluate_svm_model(
            model=svm_model, X=X_val_scaled, y=y_val, dataset_name="Validation Set"
        )

        plot_cm_matrix(
            y=y_train[train_ids],
            y_pred=train_y_pred,
            n_training_samples=training_fold.shape[0],
            save=True,
            model_name="SVM with RBF kernel",
            dataset_name="Train Set",
            filepath=f"data/plots/svm_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_SVM_train_confusionmatrix",
        )
        plot_cm_matrix(
            y=y_test,
            y_pred=test_y_pred,
            n_training_samples=training_fold.shape[0],
            save=True,
            model_name="SVM with RBF kernel",
            dataset_name="Test Set",
            filepath=f"data/plots/svm_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_SVM_test_confusionmatrix",
        )
        plot_cm_matrix(
            y=y_val,
            y_pred=val_y_pred,
            n_training_samples=training_fold.shape[0],
            save=True,
            model_name="SVM with RBF kernel",
            dataset_name="Validation Set",
            filepath=f"data/plots/svm_confusion_matrices/{config['experiment_name']}_{sample_size}_{fold}fold_SVM_val_confusionmatrix",
        )

        k_fold_metrics[sample_size]["train"].append(train_fitted_metrics)
        k_fold_metrics[sample_size]["test"].append(test_fitted_metrics)
        k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)


plot_boxplots(k_fold_metrics, sample_sizes, model="SVM")
