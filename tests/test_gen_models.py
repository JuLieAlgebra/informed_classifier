import unittest
from typing import List, Tuple

import pytest
import numpy as np

from informed_classification import generative_models


@pytest.fixture
def data() -> List[Tuple[float, generative_models.GenerativeModel]]:
    """Produces test data for testing the generative models"""
    dim = 10
    models = [
        generative_models.DisruptedModel(dim),
        generative_models.NominalModel(dim),
    ]
    test_data = [(model.sample(1), model) for model in models]
    return test_data


def test_gauss_pdf(data: List[Tuple[float, generative_models.GenerativeModel]]):
    """For input x and a gaussian process model, tests that the pdf is close"""
    for x, model in data:
        mu = model.dist.mean
        cov = model.dist.cov
        k = model.dim

        a = (2 * np.pi) ** (-k / 2)
        b = np.linalg.det(cov) ** (-1 / 2)
        c = -1 / 2 * (x - mu).T @ np.linalg.inv(cov) @ (x - mu)
        d = np.exp(c)

        # cov should not be singular or close to it
        # assert not np.isclose(np.linalg.det(cov), 0.0)

        # that our pdf function works as intended
        assert np.allclose(model.p(x), a * b * d)
