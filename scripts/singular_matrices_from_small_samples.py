import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sample_and_plot_covariance(dim, cov_matrix, n_samples):
    """Sample from a Gaussian and check if the resulting covariance is a singular matrix."""
    mean = np.zeros(dim)
    samples = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    estimated_cov = np.cov(samples, rowvar=False)

    # Check for singularity
    is_singular = np.isclose(np.linalg.det(estimated_cov), 0.0)
    return is_singular


found_singular = True
n_samples = 2
dim = 100
cov_matrix = np.eye(dim)
while found_singular:
    found_singular = sample_and_plot_covariance(
        dim=dim, cov_matrix=cov_matrix, n_samples=n_samples
    )
    n_samples += 1
print(f"Took {n_samples} to get a non-singular estimated covariance matrix ")
