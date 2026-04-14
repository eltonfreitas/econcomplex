"""
Utility functions for matrix/dataframe handling.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def pivot_to_matrix(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Convert long-format DataFrame to wide (pivot) matrix."""
    return df.pivot_table(
        index=index, columns=columns, values=values, aggfunc="sum", fill_value=fill_value
    )


def melt_matrix(
    mat: pd.DataFrame,
    index_name: str = "location",
    columns_name: str = "activity",
    values_name: str = "value",
) -> pd.DataFrame:
    """Convert wide matrix to long-format DataFrame."""
    df = mat.copy()
    # Ensure index has a name so reset_index() gives a usable column
    if df.index.name is None:
        df.index.name = index_name
    actual_index_name = df.index.name
    df.columns.name = None  # avoid extra label
    return df.reset_index().melt(
        id_vars=actual_index_name,
        var_name=columns_name,
        value_name=values_name,
    )


def validate_matrix(mat: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Return a 2-D numpy array; raise on bad input."""
    if isinstance(mat, pd.DataFrame):
        arr = mat.values.astype(float)
    elif isinstance(mat, np.ndarray):
        arr = mat.astype(float)
    else:
        arr = np.array(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2-D matrix.")
    return arr


def binarize(mat: Union[np.ndarray, pd.DataFrame], threshold: float = 1.0) -> np.ndarray:
    """Return binary matrix: 1 where mat >= threshold, else 0."""
    arr = validate_matrix(mat)
    return (arr >= threshold).astype(float)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division, returning 0 where denominator == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, 0.0)
    return result


def normalize_zscore(vec: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D vector."""
    std = np.std(vec)
    if std == 0:
        return np.zeros_like(vec)
    return (vec - np.mean(vec)) / std


def normalize_01(vec: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D vector to [0, 1]."""
    rng = vec.max() - vec.min()
    if rng == 0:
        return np.zeros_like(vec)
    return (vec - vec.min()) / rng
