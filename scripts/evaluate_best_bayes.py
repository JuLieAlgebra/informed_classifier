import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from informed_classification import bayes_classifier, generative_models, models
from informed_classification.analysis import (
    evaluate_gauss_model,
    plot_boxplots,
    plot_cm_matrix,
)
from informed_classification.common_utilities import get_config, load_data

# Evaluating on Test and Validation Sets
# _, _ = evaluate_gauss_model(true_bayes, training_fold, y_train[train_ids], "Train Set")
# _, _ = evaluate_gauss_model(true_bayes, X_test, y_test, "Test Set")
# _, _ = evaluate_gauss_model(true_bayes, X_val, y_val, "Validation Set")

config, _ = get_config()

# Load datasets
train_data = load_data(f"data/{config['experiment_name']}/train")
test_data = load_data(f"data/{config['experiment_name']}/test")
validation_data = load_data(f"data/{config['experiment_name']}/validation")

# Split features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
X_val, y_val = validation_data[:, :-1], validation_data[:, -1]

#### Evaluating True Gaussian Processes
nominal = generative_models.NominalModel(config["dim"])
disrupted = generative_models.DisruptedModel(config["dim"])
true_bayes = bayes_classifier.BayesClassifier(
    [config["ratio"], 1 - config["ratio"]], [nominal, disrupted]
)

### If I want each sample size to be 20, then I have to give kfold roughly sample*5*2
sample_sizes = [3, 5, 8, 10, 15, 20, 40, 50, 80, 100, 200, 300, 500, 800, 1000, 2000]
k_folds = 5
k_fold_metrics = {size: {"train": [], "test": [], "val": []} for size in sample_sizes}


parent_dir = "data/plots/true_gp_confusion_matrices"
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

for sample_size in sample_sizes:
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_performance = []

    sample_size_len = sample_size * 2 * k_folds
    for fold, (train_ids, val_ids) in enumerate(
        kfold.split(X=X_train[:sample_size_len], y=y_train[:sample_size_len])
    ):
        print(
            f"Each model training on approx {len(train_ids)//2} samples - Fold {fold+1}/{k_folds}"
        )

        training_fold = X_train[train_ids]
        # n_training_samples = len(train_ids)//2
        assert training_fold.shape[0] == len(train_ids)

        #### Evaluating Fitted Gaussian Processes
        print(
            "Nominal samples in train: ",
            training_fold[y_train[train_ids] == 0].shape,
            "Disrupted samples in train: ",
            training_fold[y_train[train_ids] == 1].shape,
        )

        # Evaluating on Training set
        train_fitted_metrics, train_y_pred = evaluate_gauss_model(
            true_bayes, training_fold, y_train[train_ids], dataset_name="Train Set"
        )
        # Test
        test_fitted_metrics, test_y_pred = evaluate_gauss_model(
            true_bayes, X_test, y_test, dataset_name="Test Set"
        )
        # Validation
        val_fitted_metrics, val_y_pred = evaluate_gauss_model(
            true_bayes, X_val, y_val, dataset_name="Validation Set"
        )

        k_fold_metrics[sample_size]["train"].append(train_fitted_metrics)
        k_fold_metrics[sample_size]["test"].append(test_fitted_metrics)
        k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)

        plot_cm_matrix(
            y=y_train[train_ids],
            y_pred=train_y_pred,
            n_training_samples=training_fold.shape[0],
            dataset_name="Train Set",
            save=True,
            model_name="",
            filepath=f"{parent_dir}/true_{config['experiment_name']}_{sample_size}_{fold}fold_mc_gp_train_confusionmatrix",
            title="Bayes Classifier Confusion Matrix\nWith Underlying, True Processes",
        )

        plot_cm_matrix(
            y=y_test,
            y_pred=test_y_pred,
            n_training_samples=training_fold.shape[0],
            dataset_name="Test Set",
            save=True,
            model_name="",
            filepath=f"{parent_dir}/{config['experiment_name']}_{sample_size}_{fold}fold_mc_gp_test_confusionmatrix",
            title="Bayes Classifier Confusion Matrix\nWith Underlying, True Processes",
        )

        plot_cm_matrix(
            y=y_val,
            y_pred=val_y_pred,
            n_training_samples=training_fold.shape[0],
            dataset_name="Validation Set",
            save=True,
            model_name="",
            filepath=f"{parent_dir}/true_{config['experiment_name']}_{sample_size}_{fold}fold_mc_gp_val_confusionmatrix",
            title="Bayes Classifier Confusion Matrix\nWith Underlying, True Processes",
        )


print(sample_sizes, "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
plot_boxplots(k_fold_metrics, sample_sizes, model="TrueGaussianModel", true_model=True)
