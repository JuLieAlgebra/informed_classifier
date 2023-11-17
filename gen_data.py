"""Generate the data from the generative models defined in informed_classification"""
import os
import argparse

import numpy as np
import pandas as pd

from informed_classification import generative_models


#### CONFIGURATION
dim = 20
samples = 10000
ratio = 0.7  # fraction of data that's nominal

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

#### WRITING TO FILE
if not os.path.exists("data"):
    os.makedirs("data")

filename = f"data/generated_data_{dim}.csv"
if not os.path.exists(filename):
    np.savetxt(
        f"data/generated_data_{dim}.csv",
        data,
        fmt="%.18e",
        delimiter=",",
        newline="\n",
        header="",
        footer="",
        comments="# ",
    )
