"""Preprocessing operations used by the explicit PCA pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_feature_means(features: pd.DataFrame) -> np.ndarray:
    return features.to_numpy(dtype=float).mean(axis=0)


def center_data(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean vector and centered feature matrix X_c = X - mu."""
    matrix = features.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    return means, matrix - means
