"""Transparent, manual PCA calculations using NumPy linear algebra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import center_data


@dataclass
class PCAResult:
    matrix: np.ndarray
    means: np.ndarray
    centered_matrix: np.ndarray
    covariance_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray

    def project(self, components: int) -> np.ndarray:
        """Project centered observations onto the first k principal directions."""
        return self.centered_matrix @ self.eigenvectors[:, :components]


def calculate_covariance(centered_matrix: np.ndarray) -> np.ndarray:
    """Calculate C = (1 / (n - 1)) X_c^T X_c explicitly."""
    observations = centered_matrix.shape[0]
    if observations < 2:
        raise ValueError("At least two students are required to calculate covariance.")
    return (centered_matrix.T @ centered_matrix) / (observations - 1)


def calculate_pca(matrix: np.ndarray) -> PCAResult:
    """Run the sequential PCA pipeline without using a PCA black box."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("PCA requires a 2D matrix with at least two students and two features.")
    means = matrix.mean(axis=0)
    centered = matrix - means
    covariance = calculate_covariance(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = eigenvalues.sum()
    ratios = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
    return PCAResult(
        matrix=matrix,
        means=means,
        centered_matrix=centered,
        covariance_matrix=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=ratios,
        cumulative_explained_variance=np.cumsum(ratios),
    )


def orthogonality_matrix(eigenvectors: np.ndarray) -> np.ndarray:
    return eigenvectors.T @ eigenvectors


def component_loadings(eigenvectors: np.ndarray, feature_names: list[str], components: int) -> pd.DataFrame:
    """Create a readable feature-loading table."""
    import pandas as pd

    return pd.DataFrame(
        eigenvectors[:, :components],
        index=feature_names,
        columns=[f"PC{i}" for i in range(1, components + 1)],
    )
