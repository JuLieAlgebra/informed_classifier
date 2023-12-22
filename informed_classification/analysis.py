""" Common space for analysis and plotting functions across models types """
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
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


def make_gif(
    frame_folder,
    file_pattern="",
    second_pattern=None,
    output_name="my_awesome",
    sort_files=True,
):
    image_files = []
    for image in glob.glob(f"{frame_folder}/*.png"):
        if file_pattern in image:
            if second_pattern is not None:
                if second_pattern in image:
                    image_files.append(image)
            else:
                image_files.append(image)
    # Use regular expression to extract numbers and sort by that number
    if sort_files:
        image_files.sort(key=lambda x: int(re.search(r"dis_(\d+)_", x).group(1)))

    frames = [Image.open(image) for image in image_files]
    frame_one = frames[0]
    frame_one.save(
        output_name + ".gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )


def plot_cm_matrix(
    y,
    y_pred,
    n_training_samples,
    dataset_name,
    model_name,
    save,
    filepath: str,
    title=None,
):
    # Confusion Matrix
    plt.figure()
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    if title is None:
        plt.title(
            f"Confusion Matrix\nfor {model_name} {dataset_name}\nTrained with {n_training_samples} points"
        )
    else:
        plt.title(title)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_boxplots(k_fold_metrics, sample_sizes, model, true_model=False):
    for dataset_slice in ["train", "test", "val"]:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            plt.figure(figsize=(10, 6))
            metric_data = {
                size: [
                    fold_metrics[metric]
                    for fold_metrics in k_fold_metrics[size][dataset_slice]
                ]
                for size in sample_sizes
            }
            sns.boxplot(data=pd.DataFrame(metric_data))
            if true_model is None:
                plt.title(
                    f"{model} Boxplot of {dataset_slice} {metric.capitalize()}\ntrained on different sized training sets"
                )
            else:
                plt.title(f"{model} Boxplot of {dataset_slice} {metric.capitalize()}")
            plt.xlabel("Training Set Size")
            plt.ylabel(f"{metric.capitalize()} Score")
            plt.xticks(rotation=45)
            if true_model is None:
                plt.ylim(0.4, 1.0)
            plt.xlim(left=1)
            plt.tight_layout()
            plt.savefig(
                f"data/plots/{model}_{metric}_{dataset_slice}_boxplot.png",
                bbox_inches="tight",
            )
            plt.close()


def darken_color(color, factor=0.7):
    """Darken a given RGB color."""
    return [factor * x for x in color]


def plot_multiple_boxplots(metrics_list, sample_sizes, models):
    # Define colors for each model
    model_colors = sns.color_palette("hsv", len(models))
    edge_colors = {
        model: darken_color(color) for model, color in zip(models, model_colors)
    }

    for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        plt.figure(figsize=(10, 6))
        all_metric_data = []

        # Prepare the data for all models for the specific metric
        for fold_metrics, model, color in zip(metrics_list, models, model_colors):
            for size in sample_sizes:
                metric_values = [
                    fold_metric[metric] for fold_metric in fold_metrics[size]["val"]
                ]
                all_metric_data.extend(
                    [(size, value, model, color) for value in metric_values]
                )

        plot_data = pd.DataFrame(
            all_metric_data, columns=["Sample Size", "Score", "Model", "Color"]
        )

        # Plot boxplots
        sns.boxplot(
            x="Sample Size",
            y="Score",
            hue="Model",
            data=plot_data,
            palette=dict(zip(models, model_colors)),
            dodge=True,
            linewidth=2,
            fill=False,
        )

        plt.title(f"Boxplot of {metric.capitalize()} by Sample Size and Model")
        plt.xlabel("Training Set Size")
        plt.ylabel(f"{metric.capitalize()} Score")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Adjust y-axis limits
        plt.legend(title="Model")
        plt.tight_layout()

        plt.savefig(f"data/plots/boxplot_{metric}.png", bbox_inches="tight")
        plt.close()


def plot_paramsweep_multiple_boxplots(
    metrics_list, sample_sizes, models, param_name, param_val
):
    # Define colors for each model
    model_colors = sns.color_palette("hsv", len(models))
    edge_colors = {
        model: darken_color(color) for model, color in zip(models, model_colors)
    }

    for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        plt.figure(figsize=(12, 8))
        all_metric_data = []

        # Prepare the data for all models for the specific metric
        for fold_metrics, model, color in zip(metrics_list, models, model_colors):
            for size in sample_sizes:
                metric_values = [
                    fold_metric[metric] for fold_metric in fold_metrics[size]["val"]
                ]
                all_metric_data.extend(
                    [(size, value, model, color) for value in metric_values]
                )

        plot_data = pd.DataFrame(
            all_metric_data, columns=["Sample Size", "Score", "Model", "Color"]
        )

        # Plot boxplots
        sns.boxplot(
            x="Sample Size",
            y="Score",
            hue="Model",
            data=plot_data,
            palette=dict(zip(models, model_colors)),
            dodge=True,
            linewidth=2,
            fill=False,
        )

        plt.title(
            f"Boxplot of {metric.capitalize()} by Sample Size and Model\non {param_name} equals {param_val}"
        )
        plt.xlabel("Training Set Size")
        plt.ylabel(f"{metric.capitalize()} Score")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Adjust y-axis limits
        plt.legend(title="Model")
        plt.tight_layout()

        plt.savefig(
            f"data/plots/boxplot_{metric}_{param_name}{param_val}.png",
            bbox_inches="tight",
        )
        plt.close()


def evaluate_svm_model(model, X, y, dataset_name) -> tuple[dict, np.array]:
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Metrics for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    performance = {
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return performance, y_pred


def evaluate_gauss_model(classifier, X, y, dataset_name) -> tuple[dict, np.array]:
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


def compute_nn_metrics(model, dataloader):
    model.eval()
    y_pred, y_true, y_scores = [], [], []

    with torch.no_grad():
        for X_tensor, y_tensor in dataloader:
            outputs = model(X_tensor)
            y_scores.extend(outputs.squeeze().numpy())
            y_pred.extend(outputs.squeeze().numpy() >= 0.5)
            y_true.extend(y_tensor.numpy())

    # Convert to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores),
    }
    model_outputs = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }
    return metrics, model_outputs


def roc_curve(model, X):
    y_score = model.decision_function(X)
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic for {dataset_name}")
    plt.legend(loc="lower right")
    plt.show()
