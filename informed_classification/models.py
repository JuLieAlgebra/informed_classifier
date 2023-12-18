from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss

from informed_classification.generative_models import (
    DisruptedModel,
    GenerativeModel,
    NominalModel,
)


def regularize_singular_cov(matrix: np.array):
    """We know that the covariance matrix of this problem is not degenerate/singular.
    By doing an unconstrained MLE and then clipping, or projecting, the covariance MLE
    to non-singular matrices, we can think of this as enforcing that prior.
    """
    eig_val, eig_vec = np.linalg.eigh(matrix)
    eig_val = np.clip(eig_val, a_min=1e-6, a_max=np.inf)
    return eig_vec @ np.diag(eig_val) @ eig_vec.T


class FittedGaussianModel(GenerativeModel):
    """
    Fits time-varying mean and time-varying covariance of nominal model.
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


class BayesFittedGaussianModel(GenerativeModel):
    """
    Fits time-varying mean and time-varying covariance of nominal model.
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

    def p(self, x: np.array) -> float:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> list[np.array]:
        """samples from the nominal model"""
        return self.dist.rvs(size=n).reshape((n, self.dim))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dim = 100
    n_samples = np.arange(1, 10)  # [10, 20, 50]
    for n in n_samples:
        underlying_model = NominalModel(dim=dim)
        data = underlying_model.sample(n)

        model = FittedGaussianModel(data=data)
        print(model.p(data))
        plt.plot(model.sample(10).T)
        plt.show()
