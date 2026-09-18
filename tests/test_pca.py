import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data_loader import identify_pca_columns, load_sample_data, prepare_numeric_features
from src.pca_calculator import calculate_pca, orthogonality_matrix
from src.preprocessing import center_data


def test_sample_dataset_has_expected_features():
    data = load_sample_data()
    columns = identify_pca_columns(data)
    assert len(data) == 12
    assert len(columns) == 7
    assert "Student Name" not in columns
    assert "Roll Number" not in columns


def test_centering_has_zero_feature_means():
    data = load_sample_data()
    features = prepare_numeric_features(data, identify_pca_columns(data))
    means, centered = center_data(features)
    assert np.allclose(centered.mean(axis=0), 0)
    assert np.allclose(means, features.to_numpy().mean(axis=0))


def test_manual_pca_matches_sklearn_variance():
    data = load_sample_data()
    features = prepare_numeric_features(data, identify_pca_columns(data))
    result = calculate_pca(features.to_numpy())
    reference = PCA().fit(features)
    assert np.allclose(result.eigenvalues, reference.explained_variance_)
    assert np.allclose(result.explained_variance_ratio, reference.explained_variance_ratio_)


def test_eigenvectors_are_orthonormal_and_projection_shape_is_reduced():
    matrix = np.array([[1, 2, 4], [2, 4, 1], [3, 5, 2], [4, 7, 8]], dtype=float)
    result = calculate_pca(matrix)
    assert np.allclose(orthogonality_matrix(result.eigenvectors), np.eye(3))
    assert result.project(2).shape == (4, 2)


def test_missing_values_are_imputed():
    data = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    features = prepare_numeric_features(data, ["A", "B"])
    assert not features.isna().any().any()
