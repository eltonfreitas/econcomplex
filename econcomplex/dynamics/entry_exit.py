"""
Industry entry and exit tracking across time periods.

References
----------
Balland & Rigby (2017); EconGeo (R package).
"""

import numpy as np
import pandas as pd
from typing import List, Union

from ..core.utils import validate_matrix, binarize, melt_matrix, pivot_to_matrix
from ..core.rca import rca as compute_rca


def _panel_to_matrices(
    df: pd.DataFrame, loc: str, act: str, val: str, time: str
) -> List[pd.DataFrame]:
    """Split a long-format panel into one aligned matrix per time period."""
    periods = sorted(df[time].unique())
    if len(periods) < 2:
        raise ValueError("Need at least 2 time periods.")
    mats = [pivot_to_matrix(df[df[time] == t], loc, act, val) for t in periods]
    rows = mats[0].index
    cols = mats[0].columns
    for m in mats[1:]:
        rows = rows.union(m.index)
        cols = cols.union(m.columns)
    return [m.reindex(index=rows, columns=cols, fill_value=0.0) for m in mats]


def entry(
    mats: List[Union[np.ndarray, pd.DataFrame]],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Industry entry matrix: 1 when a (region, activity) transitions from
    absent (M=0) to present (M=1) between consecutive time periods.

    Parameters
    ----------
    mats : list of array-like (R x C)
        Ordered list of value matrices (at least 2).
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    R x C entry matrix (1 = entry event occurred in at least one period).
    """
    if len(mats) < 2:
        raise ValueError("Need at least 2 matrices to detect entry.")

    is_df = isinstance(mats[0], pd.DataFrame)
    row_index = mats[0].index if is_df else None
    col_index = mats[0].columns if is_df else None

    def _bin(m):
        arr = validate_matrix(m)
        if use_rca:
            return binarize(compute_rca(arr), threshold)
        return binarize(arr, threshold)

    binary = [_bin(m) for m in mats]
    result = np.zeros_like(binary[0])

    for t in range(1, len(binary)):
        entry_t = ((binary[t - 1] == 0) & (binary[t] == 1)).astype(float)
        result = np.maximum(result, entry_t)

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=col_index)
    return result


def exit(
    mats: List[Union[np.ndarray, pd.DataFrame]],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Industry exit matrix: 1 when a (region, activity) transitions from
    present (M=1) to absent (M=0) between consecutive time periods.

    Parameters
    ----------
    mats : list of array-like (R x C)
        Ordered list of value matrices (at least 2).
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    R x C exit matrix (1 = exit event occurred in at least one period).
    """
    if len(mats) < 2:
        raise ValueError("Need at least 2 matrices to detect exit.")

    is_df = isinstance(mats[0], pd.DataFrame)
    row_index = mats[0].index if is_df else None
    col_index = mats[0].columns if is_df else None

    def _bin(m):
        arr = validate_matrix(m)
        if use_rca:
            return binarize(compute_rca(arr), threshold)
        return binarize(arr, threshold)

    binary = [_bin(m) for m in mats]
    result = np.zeros_like(binary[0])

    for t in range(1, len(binary)):
        exit_t = ((binary[t - 1] == 1) & (binary[t] == 0)).astype(float)
        result = np.maximum(result, exit_t)

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=col_index)
    return result


def entry_exit_summary(
    mats: List[Union[np.ndarray, pd.DataFrame]],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Summary of entry and exit events per (region, activity) pair.

    Returns a long-format DataFrame with columns:
    - location, activity, n_entries, n_exits, net_change

    Parameters
    ----------
    mats : list of array-like (R x C)
        Ordered list of value matrices.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    """
    if len(mats) < 2:
        raise ValueError("Need at least 2 matrices.")

    is_df = isinstance(mats[0], pd.DataFrame)
    row_index = mats[0].index if is_df else range(mats[0].shape[0])
    col_index = mats[0].columns if is_df else range(mats[0].shape[1])

    def _bin(m):
        arr = validate_matrix(m)
        if use_rca:
            return binarize(compute_rca(arr), threshold)
        return binarize(arr, threshold)

    binary = [_bin(m) for m in mats]
    n_r, n_c = binary[0].shape

    n_entries = np.zeros((n_r, n_c))
    n_exits = np.zeros((n_r, n_c))

    for t in range(1, len(binary)):
        n_entries += ((binary[t - 1] == 0) & (binary[t] == 1)).astype(float)
        n_exits += ((binary[t - 1] == 1) & (binary[t] == 0)).astype(float)

    rows = []
    for i, loc in enumerate(row_index):
        for j, act in enumerate(col_index):
            if n_entries[i, j] > 0 or n_exits[i, j] > 0:
                rows.append({
                    "location": loc,
                    "activity": act,
                    "n_entries": int(n_entries[i, j]),
                    "n_exits": int(n_exits[i, j]),
                    "net_change": int(n_entries[i, j] - n_exits[i, j]),
                })

    return pd.DataFrame(rows)


def entry_tracking(
    df: pd.DataFrame,
    loc: str,
    act: str,
    val: str,
    time: str,
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Long-format wrapper around `entry` for panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    loc, act, val, time : str
        Column names for location, activity, value, and time period.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.DataFrame with columns [loc, act, 'entry'] (entry = 1 when the
    pair entered in at least one period transition).
    """
    mats = _panel_to_matrices(df, loc, act, val, time)
    result = entry(mats, use_rca=use_rca, threshold=threshold)
    return melt_matrix(result, loc, act, "entry")


def exit_tracking(
    df: pd.DataFrame,
    loc: str,
    act: str,
    val: str,
    time: str,
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Long-format wrapper around `exit` for panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    loc, act, val, time : str
        Column names for location, activity, value, and time period.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.DataFrame with columns [loc, act, 'exit'] (exit = 1 when the
    pair exited in at least one period transition).
    """
    mats = _panel_to_matrices(df, loc, act, val, time)
    result = exit(mats, use_rca=use_rca, threshold=threshold)
    return melt_matrix(result, loc, act, "exit")
