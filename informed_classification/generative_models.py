"""Generative models"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as ss


class GenerativeModel(ABC):
    """Base class for defining a generative model"""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError(
                f"Expected dim to be non-zero positive integer, got {dim} instead."
            )
        self.dim = int(dim)

    @abstractmethod
    def p(self, x: Union[float, np.array]) -> Union[float, np.array]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> list[np.array]:
        raise NotImplementedError


class NominalModel(GenerativeModel):
    """
    Defines nominal behavior
    Is a Gaussian Process
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self._version = "2.0"
        self.dist = ss.multivariate_normal(
            mean=[self.mean(t) for t in range(self.dim)],
            cov=[
                [self.cov(t1, t2) for t1 in range(self.dim)] for t2 in range(self.dim)
            ],
            allow_singular=False,
        )

    def mean(self, t: Union[float, np.array]) -> Union[float, np.array]:
        """computes the mean function at index t"""
        return np.exp(-t / 10)

    def cov(
        self, t1: Union[float, np.array], t2: Union[float, np.array]
    ) -> Union[float, np.array]:
        """computes the covariance of t1, t2"""
        if (t1 == 0) or (t2 == 0):
            if t1 == t2:
                return 1e-6
            return 0.0
        process = 1e-3 * np.exp(-((t1 - t2) ** 2) / 100.0)
        noise = 1e-6 if t1 == t2 else 0.0
        return process + noise

    def p(self, x: Union[float, np.array]) -> Union[float, np.array]:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> np.array:
        """samples from the nominal model"""
        return self.dist.rvs(size=n).reshape((n, self.dim))


class DisruptedModel(GenerativeModel):
    """
    Defines disrupted behavior
    Is a Gaussian Process
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self._version = "2.0"
        self.dist = ss.multivariate_normal(
            mean=[self.mean(t) for t in range(self.dim)],
            cov=[
                [self.cov(t1, t2) for t1 in range(self.dim)] for t2 in range(self.dim)
            ],
            allow_singular=False,
        )

    def mean(self, t: Union[float, np.array]) -> Union[float, np.array]:
        """computes the mean function at index t"""
        return np.exp(-t / 10)

    def cov(
        self, t1: Union[float, np.array], t2: Union[float, np.array]
    ) -> Union[float, np.array]:
        """computes the covariance of t1, t2"""
        if (t1 == 0) or (t2 == 0):
            if t1 == t2:
                return 1e-6
            return 0.0
        process = 1e-3 * np.exp(-((t1 - t2) ** 2) / 100.0)
        noise = 1e-6 if t1 == t2 else 0.0
        disruption = 1e-7 * t1 * t2 + 1e-3 * np.exp(-np.sin((t1 - t2) / 2.0) ** 2 / 4.0)
        return process + noise + 1e-1 * disruption

    def p(self, x: Union[float, np.array]) -> Union[float, np.array]:
        """probability density at x"""
        return self.dist.pdf(x)

    def sample(self, n: int) -> np.array:
        """samples from the disrupted model"""
        return self.dist.rvs(size=n).reshape((n, self.dim))


if __name__ == "__main__":
    d = 100
    n = 20

    nominal = NominalModel(d)
    disrupted = DisruptedModel(d)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle("Samples from Nominal and Disrupted Time Series")
    axs[0].set_title("Nominal")
    axs[0].set_xlabel("Time")
    axs[0].plot(nominal.sample(n).T, alpha=0.5)
    axs[1].set_title("Disrupted")
    axs[1].set_xlabel("Time")
    axs[1].plot(disrupted.sample(n).T, alpha=0.5)
    plt.show()
