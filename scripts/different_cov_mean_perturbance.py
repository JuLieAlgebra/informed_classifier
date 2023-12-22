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

from informed_classification import bayes_classifier, generative_models, models
from informed_classification.analysis import (
    evaluate_gauss_model,
    evaluate_svm_model,
    plot_boxplots,
    plot_cm_matrix,
    plot_multiple_boxplots,
    plot_paramsweep_multiple_boxplots,
)
from informed_classification.common_utilities import get_config, load_data

mean_perturbances = [1e-8, 1e-7]
cov_disruption = 1e-2
dim = 100
samples = 10000
ratio = 0.5
train_test_validation_split = [0.2, 0.4, 0.4]
assert np.isclose(sum(train_test_validation_split), 1.0)

for mean_perturbance in mean_perturbances:
    #### MODELS
    disrupted = generative_models.DisruptedModel(
        dim, mean_perturbance=mean_perturbance, cov_disruption=cov_disruption
    )
    nominal = generative_models.NominalModel(dim)

    experiment_name = f"{dim}dim_{samples}samples_{ratio}ratio_{train_test_validation_split[0]}train_{train_test_validation_split[1]}test_{train_test_validation_split[2]}val_{2+mean_perturbance}nom_{2+cov_disruption}dis"
    experiment_name = experiment_name.replace(".", "_")

    #### SAMPLING
    disrupted_data = disrupted.sample(int(samples * (1 - ratio)))
    disrupted_labels = np.ones((disrupted_data.shape[0], 1))
    nominal_data = nominal.sample(int(samples * (ratio)))
    nominal_labels = np.zeros((nominal_data.shape[0], 1))

    #### PREPARING DATA TO WRITE
    x_data = np.vstack((disrupted_data, nominal_data))
    y_data = np.vstack((disrupted_labels, nominal_labels))
    data = np.hstack((x_data, y_data))

    # Ensure that the data is randomly shuffled, preserving labels with samples.
    np.random.shuffle(data)

    #### WRITING TO FILE
    if not os.path.exists("data"):
        os.makedirs("data")

    indicies = [0]
    for i in range(0, len(train_test_validation_split)):
        indicies.append(indicies[i] + int(samples * train_test_validation_split[i]))

    sections = [
        f"data/{experiment_name}/train",
        f"data/{experiment_name}/test",
        f"data/{experiment_name}/validation",
    ]
    sample_number = 0
    for i, section in enumerate(sections):
        if not os.path.exists(section):
            os.makedirs(section)
        for datapoint in data[indicies[i] : indicies[i + 1]]:
            np.save(os.path.join(section, f"generated_data_{sample_number}"), datapoint)
            sample_number += 1

    # Load datasets
    train_data = load_data(f"data/{experiment_name}/train")
    test_data = load_data(f"data/{experiment_name}/test")
    validation_data = load_data(f"data/{experiment_name}/validation")

    # Split features and labels
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    X_val, y_val = validation_data[:, :-1], validation_data[:, -1]

    # #### Evaluating True Gaussian Processes
    nominal = generative_models.NominalModel(dim=100)
    disrupted = generative_models.DisruptedModel(dim=100)
    true_bayes = bayes_classifier.BayesClassifier(
        [ratio, 1 - ratio], [nominal, disrupted]
    )

    ### If I want each sample size to be 20, then I have to give kfold roughly sample*5*2
    per_model_sample_sizes = [
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        40,
        100,
        200,
    ]  # , 300, 500]#, 800, 1000, 2000]
    sample_sizes = [s * 2 for s in per_model_sample_sizes]
    k_folds = 5
    svm_k_fold_metrics = {
        size: {"train": [], "test": [], "val": []} for size in sample_sizes
    }
    gauss_k_fold_metrics = {
        size: {"train": [], "test": [], "val": []} for size in sample_sizes
    }
    true_gaussian_k_fold_metrics = {
        size: {"train": [], "test": [], "val": []} for size in sample_sizes
    }

    scaler = StandardScaler()

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
            training_fold = X_train[train_ids]
            # n_training_samples = len(train_ids)//2
            assert training_fold.shape[0] == len(train_ids)

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
                [ratio, 1 - ratio], [fitted_nominal, fitted_disrupted]
            )

            posterior = fitted_bayes.posterior(training_fold)
            label = fitted_bayes.classify(training_fold)
            assert np.allclose(np.sum(posterior, axis=1), 1.0)

            # Evaluating on Training set
            train_fitted_metrics, train_y_pred = evaluate_gauss_model(
                fitted_bayes,
                training_fold,
                y_train[train_ids],
                dataset_name="Train Set",
            )
            # Test
            test_fitted_metrics, test_y_pred = evaluate_gauss_model(
                fitted_bayes, X_test, y_test, dataset_name="Test Set"
            )
            # Validation
            val_fitted_metrics, val_y_pred = evaluate_gauss_model(
                fitted_bayes, X_val, y_val, dataset_name="Validation Set"
            )

            gauss_k_fold_metrics[sample_size]["train"].append(train_fitted_metrics)
            gauss_k_fold_metrics[sample_size]["test"].append(test_fitted_metrics)
            gauss_k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)

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

            true_gaussian_k_fold_metrics[sample_size]["train"].append(
                train_fitted_metrics
            )
            true_gaussian_k_fold_metrics[sample_size]["test"].append(
                test_fitted_metrics
            )
            true_gaussian_k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)

            # # SVM with Gaussian RBF Kernel
            svm_model = SVC(kernel="rbf")
            svm_model.fit(X_train_scaled[train_ids], y_train[train_ids])

            if not os.path.exists("data/plots/svm_confusion_matrices"):
                os.makedirs("data/plots/svm_confusion_matrices")

            # Evaluating on SCALED Training, Test, and Validation Sets
            train_fitted_metrics, train_y_pred = evaluate_svm_model(
                model=svm_model,
                X=X_train_scaled[train_ids],
                y=y_train[train_ids],
                dataset_name="Train Set",
            )
            test_fitted_metrics, test_y_pred = evaluate_svm_model(
                model=svm_model, X=X_test_scaled, y=y_test, dataset_name="Test Set"
            )
            val_fitted_metrics, val_y_pred = evaluate_svm_model(
                model=svm_model, X=X_val_scaled, y=y_val, dataset_name="Validation Set"
            )

            svm_k_fold_metrics[sample_size]["train"].append(train_fitted_metrics)
            svm_k_fold_metrics[sample_size]["test"].append(test_fitted_metrics)
            svm_k_fold_metrics[sample_size]["val"].append(val_fitted_metrics)

    plot_paramsweep_multiple_boxplots(
        [svm_k_fold_metrics, gauss_k_fold_metrics, true_gaussian_k_fold_metrics],
        sample_sizes,
        models=["SVM", "Fitted Gaussians", "True Gaussians"],
        param_name="MeanPerturbance",
        param_val=mean_perturbance,
    )
