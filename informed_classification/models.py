from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss

from informed_classification.generative_models import (
    DisruptedModel,
    GenerativeModel,
    NominalModel,
)


def regularize_singular_cov(matrix: np.array, a_min: int = 1e-6):
    """We know that the covariance matrix of this problem is not degenerate/singular.
    By doing an unconstrained MLE and then clipping, or projecting, the covariance MLE
    to non-singular matrices, we can think of this as enforcing that prior.
    """
    eig_val, eig_vec = np.linalg.eigh(matrix)
    eig_val = np.clip(eig_val, a_min=a_min, a_max=np.inf)
    return eig_vec @ np.diag(eig_val) @ eig_vec.T


class FittedGaussianModel(GenerativeModel):
    """
    Fits mean and covariance of input data.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.
    """

    def __init__(self, data: np.array):
        self.mean_vec, self.cov_mat = self.mle(data)
        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=False,
        )
        self.dim = self.mean_vec.shape[0]

    def mle(self, data: np.array):
        if data.shape[0] == 1:
            cov_mat = np.zeros((data.shape[1], data.shape[1]))
        else:
            cov_mat = np.cov(data, rowvar=False)
        mean_vec = np.mean(data, axis=0)

        assert cov_mat.shape[0] == mean_vec.shape[0]

        return mean_vec, regularize_singular_cov(cov_mat)

    def p(self, x: np.array) -> float:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> list[np.array]:
        """samples from the nominal model"""
        return self.dist.rvs(size=n).reshape((n, self.dim))


class FittedMeanGaussianModel(FittedGaussianModel):
    """
    Fits mean of input data, uses true process's covariance.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.
    """

    def __init__(self, data: np.array, process_type: str):
        model_lookup = {"nominal": NominalModel, "disrupted": DisruptedModel}
        self.mean_vec = self.mle(data)
        self.cov_mat = model_lookup[process_type](dim=data.shape[1]).dist.cov

        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=False,
        )
        self.dim = self.mean_vec.shape[0]

    def mle(self, data: np.array):
        mean_vec = np.mean(data, axis=0)
        return mean_vec


class BayesFittedGaussianModel(FittedGaussianModel):
    """
    Fits mean and covariance of nominal model.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.
    """

    def __init__(self, data: np.array, prior: np.array):
        self.mean_vec, self.cov_mat = self.maximum_aposterior(self.mle(data), prior)

        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=True,
        )
        self.dim = self.mean_vec.shape[0]

    def maximum_aposterior(self, mle, prior):
        """ """
        pass

    def mle(self, data: np.array):
        if data.shape[0] == 1:
            cov_mat = np.zeros(data.shape[1])
        else:
            cov_mat = np.cov(data, rowvar=False)
        mean_vec = np.mean(data, axis=0)

        assert cov_mat.shape[0] == mean_vec.shape[0]

        return mean_vec, cov_mat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dim = 100
    n_samples = np.arange(1, 10, step=2)
    for n in n_samples:
        underlying_model = NominalModel(dim=dim)
        data = underlying_model.sample(n)

        model = FittedGaussianModel(data=data)
        mean_model = FittedMeanGaussianModel(data=data, process_type="nominal")

        ## Plotting the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Samples from Fitted GP Models for n={n}")

        # Plot for FittedGaussianModel
        axes[0].plot(model.sample(10).T)
        axes[0].set_title("Samples from Mean and Cov Fitted GP")
        axes[0].set_xlabel("Dimension")
        axes[0].set_ylabel("Value")

        # Plot for FittedMeanGaussianModel
        axes[1].plot(mean_model.sample(10).T)
        axes[1].set_title("Samples from Mean Fitted GP")
        axes[1].set_xlabel("Dimension")
        axes[1].set_ylabel("Value")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
