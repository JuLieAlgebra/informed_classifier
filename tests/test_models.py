import numpy as np
import pytest

from informed_classification import models


def test_non_singular_matrix():
    # Create a non-singular matrix
    matrix = np.array([[5, 2], [2, 3]])
    assert np.linalg.det(matrix) != 0  # Ensure it's non-singular

    # Process the matrix
    processed_matrix = models.regularize_singular_cov(matrix)

    # Check if the processed matrix is equal to the original matrix
    np.testing.assert_array_almost_equal(processed_matrix, matrix, decimal=6)


def test_singular_matrix():
    # Create a singular matrix
    matrix = np.array([[1, 2], [2, 4]])
    assert np.isclose(np.linalg.det(matrix), 0)  # Ensure it's singular

    # Process the matrix
    processed_matrix = models.regularize_singular_cov(matrix)

    # Check if the processed matrix is no longer singular
    assert np.linalg.det(processed_matrix) != 0
