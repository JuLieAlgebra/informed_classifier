"""Generate the data from the generative models defined in informed_classification"""
import argparse
import os

import numpy as np
import yaml

from informed_classification import generative_models
from informed_classification.common_utilities import get_config

config = get_config()
dim = config["dim"]
samples = config["samples"]
ratio = config["ratio"]
train_test_validation_split = config["train_test_validation_split"]
assert np.isclose(sum(train_test_validation_split), 1.0)

#### MODELS
disrupted = generative_models.DisruptedModel(dim)
nominal = generative_models.NominalModel(dim)

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
    f"data/{config['experiment_name']}/train",
    f"data/{config['experiment_name']}/test",
    f"data/{config['experiment_name']}/validation",
]
sample_number = 0
for i, section in enumerate(sections):
    if not os.path.exists(section):
        os.makedirs(section)
    for datapoint in data[indicies[i] : indicies[i + 1]]:
        np.save(os.path.join(section, f"generated_data_{sample_number}"), datapoint)
        sample_number += 1
