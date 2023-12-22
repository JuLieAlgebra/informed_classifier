import os

import informed_classification.analysis

parent_dir = "data/plots/confusion_matrix_gifs"
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

for section in ["test", "train", "val"]:
    informed_classification.analysis.make_gif(
        "data/plots/gp_confusion_matrices",
        f"_{section}_",
        "1fold",
        parent_dir + f"/gp_confusion_{section}",
    )

for section in ["test", "train", "val"]:
    informed_classification.analysis.make_gif(
        "data/plots/svm_confusion_matrices",
        f"_{section}_",
        "1fold",
        parent_dir + f"/svm_confusion_{section}",
    )

for section in ["nominal", "disrupted"]:
    informed_classification.analysis.make_gif(
        f"data/plots/bayes_covariance_matrix_progress/{section}/",
        "details_",
        output_name=parent_dir + f"/fitted_{section}_model_progress",
    )

for section in ["test", "train", "val"]:
    informed_classification.analysis.make_gif(
        "data/plots/true_gp_confusion_matrices",
        f"_{section}_",
        "1fold",
        parent_dir + f"/true_gp_confusion_{section}",
    )
