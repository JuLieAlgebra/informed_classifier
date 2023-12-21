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

from informed_classification import bayes_classifier, generative_models, models
from informed_classification.analysis import evaluate_gauss_model
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

#### Evaluating True Gaussian Processes
nominal = generative_models.NominalModel(config["dim"])
disrupted = generative_models.DisruptedModel(config["dim"])
true_bayes = bayes_classifier.BayesClassifier(
    [config["ratio"], 1 - config["ratio"]], [nominal, disrupted]
)

posterior = true_bayes.posterior(X_train)
label = true_bayes.classify(X_train)
assert np.allclose(np.sum(posterior, axis=1), 1.0)

# Evaluating on Test and Validation Sets
evaluate_gauss_model(true_bayes, X_train, y_train, "Train Set")
evaluate_gauss_model(true_bayes, X_test, y_test, "Test Set")
evaluate_gauss_model(true_bayes, X_val, y_val, "Validation Set")

#### Evaluating Fitted Gaussian Processes
print(
    "Nominal samples in train: ",
    X_train[y_train == 0].shape,
    "Disrupted samples in train: ",
    X_train[y_train == 1].shape,
)

fitted_nominal = models.FittedGaussianModel(data=X_train[y_train == 0])
fitted_disrupted = models.FittedGaussianModel(data=X_train[y_train == 1])
fitted_bayes = bayes_classifier.BayesClassifier(
    [config["ratio"], 1 - config["ratio"]], [fitted_nominal, fitted_disrupted]
)

posterior = fitted_bayes.posterior(X_train)
label = fitted_bayes.classify(X_train)
assert np.allclose(np.sum(posterior, axis=1), 1.0)

# Evaluating on Test and Validation Sets
evaluate_gauss_model(fitted_bayes, X_train, y_train, "Train Set")
evaluate_gauss_model(fitted_bayes, X_test, y_test, "Test Set")
evaluate_gauss_model(fitted_bayes, X_val, y_val, "Validation Set")
