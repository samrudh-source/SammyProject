"""Independent sklearn comparison for validating the manual PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .pca_calculator import PCAResult


def verify_against_sklearn(result: PCAResult) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """Compare variance and scores, aligning score signs because eigenvectors have sign ambiguity."""
    model = PCA(n_components=result.eigenvectors.shape[1])
    sklearn_scores = model.fit_transform(result.matrix)
    manual_scores = result.project(result.eigenvectors.shape[1]).copy()
    for index in range(manual_scores.shape[1]):
        if np.dot(manual_scores[:, index], sklearn_scores[:, index]) < 0:
            manual_scores[:, index] *= -1
    comparison = pd.DataFrame({
        "Component": [f"PC{i}" for i in range(1, len(result.eigenvalues) + 1)],
        "Manual eigenvalue": result.eigenvalues,
        "sklearn explained variance": model.explained_variance_,
        "Absolute difference": np.abs(result.eigenvalues - model.explained_variance_),
        "Manual EVR": result.explained_variance_ratio,
        "sklearn EVR": model.explained_variance_ratio_,
    })
    return comparison, model, manual_scores
