"""
Gini coefficient, locational Gini, and Hoover-Gini.

References
----------
Krugman (1991); Hoover & Giarratani (1985); Ellison & Glaeser (1997).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide


def gini(
    x: Union[np.ndarray, pd.Series, pd.DataFrame],
) -> Union[float, pd.Series]:
    """
    Standard Gini coefficient.

    G = (sum_{i,j} |x_i - x_j|) / (2 * n * sum_i x_i)
      = 1 - 2 * area under Lorenz curve  (computed via sorted cumulative sums)

    Parameters
    ----------
    x : 1-D array or DataFrame
        If DataFrame, computes Gini per column.

    Returns
    -------
    float (for 1-D) or pd.Series (for DataFrame).
    """
    def _gini_1d(v):
        v = np.sort(np.maximum(v, 0))
        n = len(v)
        if n == 0 or v.sum() == 0:
            return 0.0
        cumsum = np.cumsum(v)
        # Lorenz-curve area approximation
        numerator = 2 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * v.sum()
        return numerator / (n * v.sum())

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: _gini_1d(col.values), axis=0)
    if isinstance(x, pd.Series):
        return _gini_1d(x.values)
    return _gini_1d(np.array(x, dtype=float))


def locational_gini(
    mat: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Krugman Locational Gini per activity (column).

    Ranks regions by (industry-share / total-share) ratio and computes
    the area between the Lorenz curve and the 45-degree line.
    Theoretical maximum = 0.5 (full concentration).

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.

    Returns
    -------
    pd.Series indexed by activity (or ndarray).
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    total = arr.sum()
    n_r = arr.shape[0]

    col_sums = arr.sum(axis=0)  # C
    row_sums = arr.sum(axis=1)  # R

    # share of each activity per region
    s_rc = safe_divide(arr, col_sums[None, :])       # R x C  (share of region r in activity c)
    # national employment share of each region
    s_r = row_sums / total                            # R

    results = []
    for c in range(arr.shape[1]):
        # Sort regions by (s_rc / s_r) ratio
        ratio = safe_divide(s_rc[:, c], s_r)
        order = np.argsort(ratio)
        p = np.cumsum(s_r[order])
        nu = np.cumsum(s_rc[:, c][order])
        p = np.concatenate([[0], p])
        nu = np.concatenate([[0], nu])
        # Area under Lorenz curve (trapezoidal)
        area = np.trapz(nu, p)
        loc_gini = 0.5 - area
        results.append(loc_gini)

    result = np.array(results)
    if is_df:
        return pd.Series(result, index=col_index, name="locational_gini")
    return result


def hoover_gini(
    mat: Union[np.ndarray, pd.DataFrame],
    pop: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Hoover-Gini: Gini using population as horizontal axis (Hoover curve).

    Ranks regions by (industry-employment-share / population-share) ratio,
    then computes Gini-type area.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    pop : 1-D array-like (length R), optional
        Population vector. If None, uses row sums (total employment).

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if pop is None:
        pop_arr = arr.sum(axis=1)
    else:
        pop_arr = np.array(pop, dtype=float)

    total_pop = pop_arr.sum()
    col_sums = arr.sum(axis=0)  # C

    p_r = pop_arr / total_pop   # population share per region

    results = []
    for c in range(arr.shape[1]):
        if col_sums[c] == 0:
            results.append(0.0)
            continue
        e_rc = arr[:, c]
        s_rc = e_rc / col_sums[c]           # employment share of region r in activity c
        ratio = safe_divide(s_rc, p_r)
        order = np.argsort(ratio)
        p = np.cumsum(p_r[order])
        nu = np.cumsum(s_rc[order])
        p = np.concatenate([[0], p])
        nu = np.concatenate([[0], nu])
        area = np.trapz(nu, p)
        results.append(1 - 2 * area)

    result = np.array(results)
    if is_df:
        return pd.Series(result, index=col_index, name="hoover_gini")
    return result
