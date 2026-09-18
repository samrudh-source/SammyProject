"""Data loading, validation, and sample-data helpers."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd

IDENTIFIER_COLUMNS = {"student name", "student_name", "name", "roll number", "roll_number", "id"}
DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_students.csv"


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset."""
    return pd.read_csv(DEFAULT_SAMPLE_PATH)


def load_csv(source: str | Path | TextIO | BinaryIO | BytesIO | StringIO) -> pd.DataFrame:
    """Read a CSV from a path or file-like object."""
    return pd.read_csv(source)


def identify_pca_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding common student identifiers."""
    excluded = {name.casefold() for name in IDENTIFIER_COLUMNS}
    return [
        column
        for column in dataframe.columns
        if column.casefold() not in excluded and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def validate_dataset(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return PCA columns and human-readable validation warnings."""
    warnings: list[str] = []
    if dataframe.empty:
        return [], ["The dataset is empty."]
    pca_columns = identify_pca_columns(dataframe)
    if len(pca_columns) < 2:
        warnings.append("At least two numerical academic feature columns are required.")
    excluded = {name.casefold() for name in IDENTIFIER_COLUMNS}
    non_numeric = [
        column for column in dataframe.columns
        if column.casefold() not in excluded and not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        warnings.append("Ignored non-numerical columns: " + ", ".join(non_numeric))
    missing = dataframe[pca_columns].isna().sum()
    missing_columns = missing[missing > 0]
    if not missing_columns.empty:
        warnings.append(
            "Missing numerical values will be filled with the corresponding feature mean: "
            + ", ".join(f"{column} ({count})" for column, count in missing_columns.items())
        )
    return pca_columns, warnings


def prepare_numeric_features(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select numerical features and impute missing values with column means."""
    features = dataframe.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return features.fillna(features.mean()).fillna(0.0)
