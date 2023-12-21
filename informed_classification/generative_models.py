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

    def plot(self, nsamples: int = 10, title: str = ""):
        """
        Plot heat map of covariance matrix, plot n sample trajectories, and plot fitted mean
        function.
        """
        fig, axs = plt.subplots(1, 2, sharex=True)
        fig.suptitle(self.__class__.__name__, fontsize=20)

        axs[0].grid(True, alpha=0.2)
        axs[0].set_title("Process", fontsize=18)
        axs[0].set_xlabel("Time (t)", fontsize=16)
        axs[0].set_ylabel("Value (x)", fontsize=16)
        axs[0].plot(self.dist.mean, color="k", label="Mean", zorder=3)
        if nsamples > 0:
            opacity = np.clip(1.0 / nsamples, 0.01, 1.0)
            axs[0].plot([], color="b", alpha=opacity, label=f"Samples ({nsamples})")
            axs[0].plot(self.sample(nsamples).T, color="b", alpha=opacity, zorder=2)
        axs[0].legend(fontsize=14)
        axs[1].grid(False)
        axs[1].set_facecolor("k")
        axs[1].set_title("Covariance", fontsize=18)
        axs[1].set_xlabel(r"$t_j$", fontsize=16)
        axs[1].set_ylabel(r"$t_i$", fontsize=16)
        img = axs[1].imshow(
            self.dist.cov,
            cmap="bone",
            vmin=np.min(np.abs(self.dist.cov)),
            vmax=np.max(np.abs(self.dist.cov)),
            interpolation="lanczos",
        )
        fig.colorbar(img)
        plt.show()


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

    def plot(self, nsamples: int = 10, title: str = ""):
        """
        Plot heat map of covariance matrix, plot n sample trajectories, and plot fitted mean
        function.
        """
        fig, axs = plt.subplots(1, 2, sharex=True)
        fig.suptitle(self.__class__.__name__, fontsize=20)

        axs[0].grid(True, alpha=0.2)
        axs[0].set_title("Process", fontsize=18)
        axs[0].set_xlabel("Time (t)", fontsize=16)
        axs[0].set_ylabel("Value (x)", fontsize=16)
        axs[0].plot(self.dist.mean, color="k", label="Mean", zorder=3)
        if nsamples > 0:
            opacity = np.clip(1.0 / nsamples, 0.01, 1.0)
            axs[0].plot([], color="b", alpha=opacity, label=f"Samples ({nsamples})")
            axs[0].plot(self.sample(nsamples).T, color="b", alpha=opacity, zorder=2)
        axs[0].legend(fontsize=14)
        axs[1].grid(False)
        axs[1].set_facecolor("k")
        axs[1].set_title("Covariance", fontsize=18)
        axs[1].set_xlabel(r"$t_j$", fontsize=16)
        axs[1].set_ylabel(r"$t_i$", fontsize=16)
        img = axs[1].imshow(
            self.dist.cov,
            cmap="bone",
            vmin=np.min(np.abs(self.dist.cov)),
            vmax=np.max(np.abs(self.dist.cov)),
            interpolation="lanczos",
        )
        fig.colorbar(img)
        plt.show()


if __name__ == "__main__":
    d = 100
    n = 20

    nominal = NominalModel(d)
    disrupted = DisruptedModel(d)

    # Extra information plots about each ground-truth process
    nominal.plot(
        title="Ground-Truth Nominal Model\nMean function, sample trajectories, and covariance function"
    )
    disrupted.plot(
        title="Ground-Truth Disrupted Model\nMean function, sample trajectories, and covariance function"
    )

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle("Samples from Nominal and Disrupted Time Series")
    axs[0].set_title("Nominal")
    axs[0].set_xlabel("Time")
    axs[0].plot(nominal.sample(n).T, alpha=0.5)
    axs[1].set_title("Disrupted")
    axs[1].set_xlabel("Time")
    axs[1].plot(disrupted.sample(n).T, alpha=0.5)
    plt.show()
