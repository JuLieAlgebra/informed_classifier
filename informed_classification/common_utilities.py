import argparse
import os

import numpy as np
import yaml

from informed_classification import generative_models


def cli():
    parser = argparse.ArgumentParser(description="SVM Training Script")
    parser.add_argument(
        "--config", help="YAML config file name in config/ dir.", required=True
    )
    args = parser.parse_args()
    return args


def get_config(config_filename: str = None) -> tuple[dict, str]:
    if config_filename is None:
        args = cli()
        config_filename = "config/" + args.config
    with open(config_filename, "r") as file:
        config = yaml.safe_load(file)

    experiment_name = f"{config['dim']}dim_{config['samples']}samples_{config['ratio']}ratio_{config['train_test_validation_split'][0]}train_{config['train_test_validation_split'][1]}test_{config['train_test_validation_split'][2]}val_{config['nominal_version']}nom_{config['disrupted_version']}dis"
    experiment_name = experiment_name.replace(".", "_")
    config["experiment_name"] = experiment_name

    return config, config_filename


def update_experiment_name(
    experiment_name: str,
    nominal_model: generative_models.NominalModel,
    disrupted_model: generative_models.DisruptedModel,
) -> str:
    nominal_version = nominal_model._version.replace(".", "_")
    disrupted_version = disrupted_model._version.replace(".", "_")

    temp = experiment_name.split("val_")
    experiment_name = temp[0] + f"val_{nominal_version}nom_{disrupted_version}dis"
    return experiment_name


def load_data(section: str) -> np.array:
    data = []
    for filename in os.listdir(section):
        if filename.endswith(".npy"):
            datapoint = np.load(os.path.join(section, filename))
            data.append(datapoint)
    return np.array(data)
