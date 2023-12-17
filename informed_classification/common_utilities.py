import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description="SVM Training Script")
    parser.add_argument(
        "--config", help="YAML config file name in config/ dir.", required=True
    )
    args = parser.parse_args()

    with open("config/" + args.config, "r") as file:
        config = yaml.safe_load(file)

    config[
        "experiment_name"
    ] = f"{config['dim']}dim_{config['samples']}samples_{config['ratio']}ratio_{config['train_test_validation_split'][0]}train_{config['train_test_validation_split'][1]}test_{config['train_test_validation_split'][2]}val"

    return config


def load_data(section):
    data = []
    for filename in os.listdir(section):
        if filename.endswith(".npy"):
            datapoint = np.load(os.path.join(section, filename))
            data.append(datapoint)
    return np.array(data)
