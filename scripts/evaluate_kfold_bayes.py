import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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
from informed_classification.common_utilities import get_config, load_data


def evaluate_gauss_model(
    classifier, X, y, dataset_name, filepath: str = None
) -> tuple[dict, np.array]:
    y_pred = classifier.classify(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    performance = {
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(f"Metrics for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return performance, y_pred


def plot_cm_matrix(
    y, y_pred, n_training_samples, dataset_name, model_name, save, filepath: str
):
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(
        f"Confusion Matrix for {model_name} {dataset_name}\\ with {n_training_samples} training points"
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig(filepath)
    else:
        plt.show()


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
sample_sizes = [20, 40, 50, 80, 100, 200, 300, 500, 800, 1000, 2000]
k_folds = 5
k_fold_metrics = {s: None for s in sample_sizes}
for sample_size in sample_sizes:
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_performance = []

    sample_size_len = sample_size * 2 * k_folds
    for fold, (train_ids, val_ids) in enumerate(
        kfold.split(X=X_train[:sample_size_len], y=y_train[:sample_size_len])
    ):
        print(f"Training on {sample_size} samples - Fold {fold+1}/{k_folds}")

        training_fold = X_train[train_ids]

        # Evaluating on Test and Validation Sets
        # evaluate_gauss_model(true_bayes, training_fold, y_train[train_ids], "Train Set")
        # evaluate_gauss_model(true_bayes, X_test, y_test, "Test Set")
        # evaluate_gauss_model(true_bayes, X_val, y_val, "Validation Set")

        #### Evaluating Fitted Gaussian Processes
        print(
            "Nominal samples in train: ",
            training_fold[y_train[train_ids] == 0].shape,
            "Disrupted samples in train: ",
            training_fold[y_train[train_ids] == 1].shape,
        )

        fitted_nominal = models.FittedGaussianModel(
            data=training_fold[y_train[train_ids] == 0]
        )
        fitted_disrupted = models.FittedGaussianModel(
            data=training_fold[y_train[train_ids] == 1]
        )
        fitted_bayes = bayes_classifier.BayesClassifier(
            [config["ratio"], 1 - config["ratio"]], [fitted_nominal, fitted_disrupted]
        )

        posterior = fitted_bayes.posterior(training_fold)
        label = fitted_bayes.classify(training_fold)
        assert np.allclose(np.sum(posterior, axis=1), 1.0)

        # Evaluating on Test and Validation Sets
        train_fitted_metrics, train_y_pred = evaluate_gauss_model(
            fitted_bayes, training_fold, y_train[train_ids], "Train Set"
        )
        plot_cm_matrix(
            y=y_train[train_ids],
            y_pred=train_y_pred,
            n_training_samples=training_fold.shape[1],
            dataset_name="Train Set",
            save=True,
            model_name="Process A, B fitted mean and cov Gaussian Process",
            filepath=f"data/plots/{config['experiment_name']}_mc_gp_train_confusionmatrix",
        )
        test_fitted_metrics, test_y_pred = evaluate_gauss_model(
            fitted_bayes, X_test, y_test, "Test Set"
        )
        plot_cm_matrix(
            y=X_test,
            y_pred=y_test,
            n_training_samples=training_fold.shape[1],
            dataset_name="Test Set",
            save=True,
            model_name="Process A, B fitted mean and cov Gaussian Process",
            filepath=f"data/plots/{config['experiment_name']}_mc_gp_test_confusionmatrix",
        )
        val_fitted_metrics, val_y_pred = evaluate_gauss_model(
            fitted_bayes, X_val, y_val, "Validation Set"
        )
        plot_cm_matrix(
            y=X_val,
            y_pred=y_val,
            n_training_samples=training_fold.shape[1],
            dataset_name="Validation Set",
            save=True,
            model_name="Process A, B fitted mean and cov Gaussian Process",
            filepath=f"data/plots/{config['experiment_name']}_mc_gp_val_confusionmatrix",
        )
