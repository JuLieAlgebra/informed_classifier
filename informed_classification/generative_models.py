"""Generative models"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy import stats as ss
from matplotlib import pyplot as plt


class GenerativeModel(ABC):
    """Base class for defining a generative model"""

    def __init__(self, dim: int):
        self.dim = int(dim)

    @abstractmethod
    def p(x: np.array) -> float:
        raise NotImplementedError

    @abstractmethod
    def sample(n: int) -> List[np.array]:
        raise NotImplementedError


class NominalModel(GenerativeModel):
    """
    Defines nominal behavior
    Is a Gaussian Process
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.dist = ss.multivariate_normal(
            mean=[self.mean(t) for t in range(self.dim)],
            cov=[
                [self.cov(t1, t2) for t1 in range(self.dim)] for t2 in range(self.dim)
            ],
        )

    def mean(self, t: float) -> float:
        """computes the mean function at index t"""
        return np.exp(-t / 10)

    def cov(self, t1: float, t2: float) -> float:
        """computes the covariance of t1, t2"""
        return 0.01 * np.exp(-((t1 - t2) ** 2) / 8)

    def p(self, x: np.array) -> float:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> List[np.array]:
        """samples from the nominal model"""
        return self.dist.rvs(size=n)


class DisruptedModel(GenerativeModel):
    """
    Defines disrupted behavior
    Is a Gaussian Process
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.dist = ss.multivariate_normal(
            mean=[self.mean(t) for t in range(self.dim)],
            cov=[
                [self.cov(t1, t2) for t1 in range(self.dim)] for t2 in range(self.dim)
            ],
        )

    def mean(self, t: float) -> float:
        """computes the mean function at index t"""
        return np.cos(t / 5) * np.exp(-t / 20)

    def cov(self, t1: float, t2: float) -> float:
        """computes the covariance of t1, t2"""
        return 0.01 * np.exp(-((t1 - t2) ** 2) / 8)

    def p(self, x: np.array) -> float:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> List[np.array]:
        """samples from the disrupted model"""
        return self.dist.rvs(size=n)


class CustomDistribution(GenerativeModel):
    """
    Combines a chi-square and multivariate Gaussian distribution
    """

    def __init__(self, dim: int, df: int, mean: List[float], cov: List[List[float]]):
        super().__init__(dim)
        self.df = df
        self.mean = mean
        self.cov = cov
        self.dist_chi2 = ss.chi2(df=self.df)
        self.dist_gaussian = ss.multivariate_normal(mean=self.mean, cov=self.cov)

    def p(self, x: np.array) -> float:
        """Probability density at x"""
        chi_square_pdf = self.dist_chi2.pdf(x[0])
        gaussian_pdf = self.dist_gaussian.pdf(x)
        return chi_square_pdf * gaussian_pdf

    @classmethod
    def likelihood(self, k: int, mean: np.array, cov: np.array, x: np.array):
        chi = ss.chi2(df=k)
        gauss = ss.multivariate_normal(mean=mean, cov=cov)
        return chi.pdf(x[0]) * gauss.pdf(x)

    def sample(self, n: int) -> np.array:
        """Sample from the custom distribution"""
        chi_square_samples = self.dist_chi2.rvs(size=n)
        gaussian_samples = self.dist_gaussian.rvs(size=n)
        return np.column_stack((chi_square_samples, gaussian_samples))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = 100
    n = 10

    nominal = NominalModel(d)
    disrupted = DisruptedModel(d)

    # Create an instance of the custom distribution
    custom_dist = CustomDistribution(dim=d, df=3, mean=[1, 2], cov=[[1, 0.5], [0.5, 2]])

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)  # Increase the number of subplots to 3
    fig.suptitle("Samples")
    
    axs[0].set_title("Nominal")
    axs[0].plot(nominal.sample(n).T, alpha=0.5)

    axs[1].set_title("Disrupted")
    axs[1].plot(disrupted.sample(n).T, alpha=0.5)

    axs[2].set_title("Custom")
    axs[2].plot(np.array(custom_dist.sample(n)).T, alpha=0.5)

    plt.show()
