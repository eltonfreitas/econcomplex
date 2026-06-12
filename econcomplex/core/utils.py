"""
Utility functions for matrix/dataframe handling.
"""

import numpy as np
import pandas as pd
from typing import Union


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


def make_sample_data(
    n_locs: int = 50,
    n_acts: int = 30,
    seed: int = 42,
    loc_col: str = "loc",
    act_col: str = "act",
    val_col: str = "val",
) -> pd.DataFrame:
    """
    Generate synthetic long-format data for examples and testing.

    The data has the nested (triangular) structure typical of real
    location-activity matrices: high-capability locations are active in
    many activities (including complex ones), while low-capability
    locations concentrate on ubiquitous activities. This makes the
    resulting ECI/PCI, proximity, and density values meaningful.

    Parameters
    ----------
    n_locs : int
        Number of locations (rows).
    n_acts : int
        Number of activities (columns).
    seed : int
        Random seed for reproducibility.
    loc_col, act_col, val_col : str
        Column names of the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Long format with columns [loc_col, act_col, val_col],
        containing only positive entries.
    """
    rng = np.random.default_rng(seed)

    # Capability of each location and requirement of each activity
    capability = np.sort(rng.uniform(0.1, 1.0, n_locs))[::-1]
    requirement = np.sort(rng.uniform(0.0, 0.9, n_acts))

    # Presence probability falls with the capability gap (nested structure)
    gap = capability[:, None] - requirement[None, :]
    presence = rng.random((n_locs, n_acts)) < 1 / (1 + np.exp(-10 * gap))

    values = presence * rng.lognormal(mean=3.0, sigma=1.0, size=(n_locs, n_acts))

    locs = [f"L{i + 1:03d}" for i in range(n_locs)]
    acts = [f"A{j + 1:03d}" for j in range(n_acts)]

    df = pd.DataFrame(values, index=locs, columns=acts)
    long = melt_matrix(df, loc_col, act_col, val_col)
    long = long[long[val_col] > 0].reset_index(drop=True)
    return long
