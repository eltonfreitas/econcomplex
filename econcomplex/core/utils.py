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


def validate_matrix(
    mat: Union[np.ndarray, pd.DataFrame],
    nonnegative: bool = True,
) -> np.ndarray:
    """
    Return a 2-D float array; raise a clear error on bad input.

    Rejects NaN/infinite cells and, unless ``nonnegative=False``, negative
    values: every indicator of the library is defined for non-negative
    quantities (employment, exports, counts, RCA).
    """
    if isinstance(mat, pd.DataFrame):
        arr = mat.values.astype(float)
    elif isinstance(mat, np.ndarray):
        arr = mat.astype(float)
    else:
        arr = np.array(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2-D matrix.")
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        raise ValueError(
            f"Input matrix has {n_bad} NaN or infinite cell(s). Fill them "
            "(e.g. `mat.fillna(0)`) or drop the affected rows/columns first."
        )
    if nonnegative and (arr < 0).any():
        raise ValueError(
            f"Input matrix has {int((arr < 0).sum())} negative cell(s); the "
            "indicators are defined for non-negative values."
        )
    return arr


def align_vector(vec, labels, n: int, name: str = "vector",
                 allow_missing: bool = False) -> np.ndarray:
    """
    Return `vec` as a float array ordered like `labels`.

    A labelled ``pd.Series`` is matched to the matrix by label, so its
    order does not matter and extra labels are ignored. Lists, arrays and
    Series with a default integer index are taken positionally and must
    have length `n`. Labels of the matrix absent from `vec` raise an
    error unless ``allow_missing=True`` (they become NaN).
    """
    by_label = (
        isinstance(vec, pd.Series) and labels is not None
        and not (isinstance(vec.index, pd.RangeIndex)
                 and not isinstance(labels, pd.RangeIndex))
    )
    if by_label:
        if vec.index.has_duplicates:
            raise ValueError(f"{name} has duplicated labels.")
        missing = pd.Index(labels).difference(vec.index)
        if len(missing) and not allow_missing:
            raise ValueError(
                f"{name} is missing {len(missing)} label(s) of the matrix, "
                f"e.g. {list(missing[:3])}."
            )
        return vec.reindex(labels).to_numpy(dtype=float)
    arr = np.asarray(vec, dtype=float).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"Length of {name} ({arr.shape[0]}) must be {n}.")
    return arr


def align_square(mat, labels, n: int, name: str = "phi") -> np.ndarray:
    """
    Return a square matrix (e.g. a proximity matrix) as a float array with
    rows and columns ordered like `labels`. DataFrames are matched by
    label; arrays are taken positionally and must be n x n.
    """
    if isinstance(mat, pd.DataFrame) and labels is not None:
        missing = pd.Index(labels).difference(mat.index.intersection(mat.columns))
        if len(missing):
            raise ValueError(
                f"{name} is missing {len(missing)} activity label(s) of the "
                f"matrix, e.g. {list(missing[:3])}."
            )
        return mat.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"{name} must be {n} x {n}; got {arr.shape}.")
    return arr


def align_matrix(mat, rows, cols, name: str = "matrix") -> np.ndarray:
    """
    Return an R x C matrix as a float array ordered like `rows` and `cols`.
    DataFrames are matched by label (cells absent from `mat` become NaN);
    arrays are taken positionally and must have the right shape.
    """
    if isinstance(mat, pd.DataFrame):
        return mat.reindex(index=rows, columns=cols).to_numpy(dtype=float)
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (len(rows), len(cols)):
        raise ValueError(f"{name} must be {(len(rows), len(cols))}; got {arr.shape}.")
    return arr


def binarize(mat: Union[np.ndarray, pd.DataFrame], threshold: float = 1.0) -> np.ndarray:
    """Return binary matrix: 1 where mat >= threshold, else 0."""
    arr = validate_matrix(mat, nonnegative=False)
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

    Examples
    --------
    >>> import econcomplex as ec
    >>> df = ec.make_sample_data(n_locs=30, n_acts=15, seed=42)
    >>> df.columns.tolist()
    ['loc', 'act', 'val']
    >>> ec.pivot_to_matrix(df, "loc", "act", "val").shape
    (30, 15)
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
