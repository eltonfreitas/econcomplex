"""
Concentration and dispersion measures.

References
----------
Herfindahl-Hirschman (1945/1950); Shannon (1948); Hoover (1936).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide


def herfindahl(
    mat: Union[np.ndarray, pd.DataFrame],
    normalize: bool = True,
) -> Union[pd.Series, np.ndarray]:
    """
    Herfindahl-Hirschman Index (HHI) per region.

    HHI_r = sum_c (x_{rc} / X_r)^2

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    normalize : bool
        If True, normalize to [0, 1] using (HHI - 1/C) / (1 - 1/C).

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)
    hhi = (shares ** 2).sum(axis=1)

    if normalize:
        n = arr.shape[1]
        min_hhi = 1.0 / n
        hhi = safe_divide(hhi - min_hhi, 1.0 - min_hhi)

    if is_df:
        return pd.Series(hhi, index=row_index, name="herfindahl")
    return hhi


def shannon_entropy(
    mat: Union[np.ndarray, pd.DataFrame],
    base: float = 2.0,
) -> Union[pd.Series, np.ndarray]:
    """
    Shannon entropy per region.

    H_r = -sum_c (s_{rc} * log_base(s_{rc}))
    where s_{rc} = x_{rc} / X_r

    Higher entropy = more diversified regional economy.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    base : float
        Logarithm base (default 2 = bits).

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_shares = np.where(shares > 0, np.log(shares) / np.log(base), 0.0)

    result = -(shares * log_shares).sum(axis=1)

    if is_df:
        return pd.Series(result, index=row_index, name="shannon_entropy")
    return result


def hoover_index(
    mat: Union[np.ndarray, pd.DataFrame],
    pop: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Hoover Index (Robin Hood Index) per activity.

    H_c = (1/2) * sum_r |E_r/E_total - A_r/A_total| * 100

    where E_r = employment of region r in activity c,
          A_r = total employment / population of region r.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    pop : array-like (length R), optional
        Population/reference vector. If None, uses row sums.

    Returns
    -------
    pd.Series indexed by activity (values 0–100).
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if pop is None:
        pop_arr = arr.sum(axis=1)
    else:
        pop_arr = np.array(pop, dtype=float)

    total = arr.sum()
    total_pop = pop_arr.sum()

    col_sums = arr.sum(axis=0)

    results = []
    for c in range(arr.shape[1]):
        if col_sums[c] == 0:
            results.append(0.0)
            continue
        share_e = arr[:, c] / col_sums[c]
        share_a = pop_arr / total_pop
        hi = 0.5 * np.abs(share_e - share_a).sum() * 100
        results.append(hi)

    result = np.array(results)
    if is_df:
        return pd.Series(result, index=col_index, name="hoover_index")
    return result
