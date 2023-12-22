from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ss

from informed_classification.generative_models import (
    DisruptedModel,
    GenerativeModel,
    NominalModel,
)


def regularize_singular_cov(matrix: np.array, a_min: float = 1e-6) -> np.array:
    """We know that the covariance matrix of this problem is not degenerate/singular.
    By doing an unconstrained MLE and then clipping, or projecting, the covariance MLE
    to non-singular matrices, we can think of this as enforcing that prior.
    """
    eig_val, eig_vec = np.linalg.eigh(matrix)
    eig_val = np.clip(eig_val, a_min=a_min, a_max=np.inf)
    return eig_vec @ np.diag(eig_val) @ eig_vec.T


class FittedGaussianModel(GenerativeModel):
    """
    Fits mean vector and covariance matrix of input data.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.

    Inputs:
        data: Assumed to be (n_samples, time_dimension) with no labels included.
    """

    def __init__(self, data: np.array):
        self.data_shape = data.shape
        self.mean_vec, self.cov_mat = self.mle(data)
        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=False,
        )
        self.dim = self.mean_vec.shape[0]

    def mle(self, data: np.array) -> tuple[np.array, np.array]:
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

    def plot(
        self,
        nsamples: int = 10,
        title: str = "",
        save=True,
        filepath="data/plots/figure",
    ):
        """
        Plot heat map of covariance matrix, plot n sample trajectories, and plot fitted mean
        function.
        """
        fig, axs = plt.subplots(1, 2, sharex=True)
        fig.suptitle(
            f"{self.__class__.__name__} trained on {self.data_shape[0]} samples",
            fontsize=20,
        )

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
        if save:
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class FittedMeanGaussianModel(FittedGaussianModel):
    """
    Fits mean vector of input data, uses true process's covariance matrix.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.

    Inputs:
        data: Assumed to be (n_samples, time_dimension) with no labels included.
        process_type: Either 'nominal' or 'disrupted'. Defines which covariance matrix
                      is assumed to be known and will be used.
    """

    def __init__(self, data: np.array, process_type: str):
        self.data_shape = data.shape
        model_lookup = {"nominal": NominalModel, "disrupted": DisruptedModel}
        self.mean_vec = self.mle(data)
        self.cov_mat = model_lookup[process_type](dim=data.shape[1]).dist.cov

        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=False,
        )
        self.dim = self.mean_vec.shape[0]

    def mle(self, data: np.array) -> np.array:
        mean_vec = np.mean(data, axis=0)
        return mean_vec


class FittedCovGaussianModel(FittedGaussianModel):
    """
    Fits covariance matrix of input data, uses true process's mean vector.
    Assumes underlying process is a Gaussian Process/Multivariate Normal.

    Inputs:
        data: Assumed to be (n_samples, time_dimension) with no labels included.
        process_type: Either 'nominal' or 'disrupted'. Defines which mean vector
                      is assumed to be known and will be used.
    """

    def __init__(self, data: np.array, process_type: str):
        self.data_shape = data.shape
        model_lookup = {"nominal": NominalModel, "disrupted": DisruptedModel}
        self.mean_vec = model_lookup[process_type](dim=data.shape[1]).dist.mean
        self.cov_mat = self.mle(data)

        self.dist = ss.multivariate_normal(
            mean=self.mean_vec,
            cov=self.cov_mat,
            allow_singular=False,
        )
        self.dim = self.mean_vec.shape[0]

    def mle(self, data: np.array) -> np.array:
        if data.shape[0] == 1:
            cov_mat = np.zeros((data.shape[1], data.shape[1]))
        else:
            cov_mat = np.cov(data, rowvar=False)

        return regularize_singular_cov(cov_mat)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dim = 100
    n_samples = np.arange(1, 10, step=2)
    for n in n_samples:
        underlying_model = NominalModel(dim=dim)
        data = underlying_model.sample(n)

        model = FittedGaussianModel(data=data)
        mean_nominal_model = FittedMeanGaussianModel(data=data, process_type="nominal")
        mean_disrupted_model = FittedMeanGaussianModel(
            data=data, process_type="disrupted"
        )
        cov_nominal_model = FittedCovGaussianModel(data=data, process_type="nominal")
        cov_disrupted_model = FittedCovGaussianModel(
            data=data, process_type="disrupted"
        )

        models = [
            model,
            mean_nominal_model,
            mean_disrupted_model,
            cov_nominal_model,
            cov_disrupted_model,
        ]
        for m in models:
            m.plot(nsamples=5)

        ## Plotting the results
        fig, axes = plt.subplots(1, 5, figsize=(23, 5))
        fig.suptitle(f"Samples from Fitted GP Models for n={n}")

        # FittedGaussianModel
        axes[0].plot(model.sample(10).T)
        axes[0].set_title("Mean and Cov Fitted GP")
        axes[0].set_xlabel("Dimension")
        axes[0].set_ylabel("Value")

        # FittedMeanGaussianModel (Nominal)
        axes[1].plot(mean_nominal_model.sample(10).T)
        axes[1].set_title("Mean Fitted GP (Nominal)")
        axes[1].set_xlabel("Dimension")

        # FittedMeanGaussianModel (Disrupted)
        axes[2].plot(mean_disrupted_model.sample(10).T)
        axes[2].set_title("Mean Fitted GP (Disrupted)")
        axes[2].set_xlabel("Dimension")

        # FittedCovGaussianModel (Nominal)
        axes[3].plot(cov_nominal_model.sample(10).T)
        axes[3].set_title("Cov Fitted GP (Nominal)")
        axes[3].set_xlabel("Dimension")

        # FittedCovGaussianModel (Disrupted)
        axes[4].plot(cov_disrupted_model.sample(10).T)
        axes[4].set_title("Cov Fitted GP (Disrupted)")
        axes[4].set_xlabel("Dimension")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
