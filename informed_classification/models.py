from abc import ABC, abstractmethod

import numpy as np
from generative_models import DisruptedModel, GenerativeModel, NominalModel
from scipy import stats as ss


class FittedGaussianModel(GenerativeModel):
    """
    Fits time-varying mean and time-varying covariance of nominal model.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.
    """

    def __init__(self, data: np.array):
        self.mean_vec, self.cov_mat = self.mle(data)
        print(self.mean_vec)
        print(self.cov_mat)
        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=True,
        )
        self.dim = self.mean_vec.shape[0]

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
