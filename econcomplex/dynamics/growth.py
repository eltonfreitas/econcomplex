"""
Regional and industry growth rates.
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, safe_divide, pivot_to_matrix


def growth_rate(
    mat1: Union[np.ndarray, pd.DataFrame],
    mat2: Union[np.ndarray, pd.DataFrame],
    axis: int = 0,
    pct: bool = True,
) -> Union[pd.Series, np.ndarray]:
    """
    Growth rate between two time periods.

    g = (sum_t2 - sum_t1) / sum_t1  [* 100 if pct]

    Parameters
    ----------
    mat1 : array-like (R x C)
        Matrix for time period 1.
    mat2 : array-like (R x C)
        Matrix for time period 2 (same shape as mat1).
    axis : int
        0 = aggregate over columns (regional growth), rows returned.
        1 = aggregate over rows (industry growth), columns returned.
    pct : bool
        If True, multiply by 100 to return percentage.

    Returns
    -------
    pd.Series or ndarray.
    """
    is_df = isinstance(mat1, pd.DataFrame)

    arr1 = validate_matrix(mat1)
    arr2 = validate_matrix(mat2)

    if arr1.shape != arr2.shape:
        raise ValueError("mat1 and mat2 must have the same shape.")

    sum1 = arr1.sum(axis=1 - axis)  # opposite axis
    sum2 = arr2.sum(axis=1 - axis)

    g = safe_divide(sum2 - sum1, sum1)
    if pct:
        g = g * 100

    if is_df:
        index = mat1.index if axis == 0 else mat1.columns
        name = "growth_region" if axis == 0 else "growth_industry"
        return pd.Series(g, index=index, name=name)
    return g


def growth_matrix(
    mat1: Union[np.ndarray, pd.DataFrame],
    mat2: Union[np.ndarray, pd.DataFrame],
    pct: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Element-wise growth rate matrix.

    g_{rc} = (x_{rc,t2} - x_{rc,t1}) / x_{rc,t1}

    Parameters
    ----------
    mat1, mat2 : array-like (R x C)
    pct : bool
        Multiply by 100.

    Returns
    -------
    R x C growth rate matrix.
    """
    is_df = isinstance(mat1, pd.DataFrame)
    row_index = mat1.index if is_df else None
    col_index = mat1.columns if is_df else None

    arr1 = validate_matrix(mat1)
    arr2 = validate_matrix(mat2)

    if arr1.shape != arr2.shape:
        raise ValueError("mat1 and mat2 must have the same shape.")

    g = safe_divide(arr2 - arr1, arr1)
    if pct:
        g = g * 100

    if is_df:
        return pd.DataFrame(g, index=row_index, columns=col_index)
    return g


def growth_rates(
    df: pd.DataFrame,
    loc: str,
    act: str,
    val: str,
    time: str,
    axis: int = 0,
    pct: bool = True,
) -> pd.DataFrame:
    """
    Long-format wrapper around `growth_rate` for panel data.

    Computes the growth rate between each pair of consecutive time periods.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    loc, act, val, time : str
        Column names for location, activity, value, and time period.
    axis : int
        0 = growth per location, 1 = growth per activity.
    pct : bool
        If True, return percentage.

    Returns
    -------
    pd.DataFrame with columns [unit, time, 'growth'], where unit is `loc`
    (axis=0) or `act` (axis=1) and `time` marks the end of each interval.
    """
    periods = sorted(df[time].unique())
    if len(periods) < 2:
        raise ValueError("Need at least 2 time periods to compute growth.")

    unit_col = loc if axis == 0 else act
    out = []
    for t1, t2 in zip(periods[:-1], periods[1:]):
        m1 = pivot_to_matrix(df[df[time] == t1], loc, act, val)
        m2 = pivot_to_matrix(df[df[time] == t2], loc, act, val)
        rows = m1.index.union(m2.index)
        cols = m1.columns.union(m2.columns)
        m1 = m1.reindex(index=rows, columns=cols, fill_value=0.0)
        m2 = m2.reindex(index=rows, columns=cols, fill_value=0.0)
        g = growth_rate(m1, m2, axis=axis, pct=pct)
        out.append(pd.DataFrame({unit_col: g.index, time: t2, "growth": g.values}))

    return pd.concat(out, ignore_index=True)
