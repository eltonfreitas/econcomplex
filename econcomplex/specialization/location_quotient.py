"""
Location quotient (LQ) variants and specialization indices.

References
----------
Hoover & Giarratani (1985); Hachman (various); Balland & Rigby (2017).
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, safe_divide
from ..core.rca import rca as compute_rca


def location_quotient(
    mat: Union[np.ndarray, pd.DataFrame],
    binary: bool = False,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Location Quotient (identical to RCA / Balassa Index).

    LQ_{rc} = (x_{rc}/X_r) / (X_c/X_total)

    Alias of `core.rca.rca`.
    """
    return compute_rca(mat, binary=binary, threshold=threshold)


def location_quotient_avg(
    mat: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Weighted average LQ per region (Coefficient of Specialization, Hoover 1985).

    avg_LQ_r = sum_c (LQ_{rc} * s_{rc})
    where s_{rc} = x_{rc} / X_r  (share of activity c in region r)

    A value > 1 indicates region is more specialized than the nation average.

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    lq = compute_rca(arr)
    if isinstance(lq, pd.DataFrame):
        lq = lq.values

    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)  # s_{rc}

    result = (lq * shares).sum(axis=1)

    if is_df:
        return pd.Series(result, index=row_index, name="lq_avg")
    return result


def hachman_index(
    mat: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Hachman Index (structural similarity to national economy).

    H_r = 1 / avg_LQ_r

    Ranges 0 to 1; value of 1 means the regional economy perfectly
    mirrors the national structure.

    Returns
    -------
    pd.Series indexed by region (clipped to [0, 1]).
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    avg_lq = location_quotient_avg(mat)
    if isinstance(avg_lq, pd.Series):
        avg_lq_arr = avg_lq.values
    else:
        avg_lq_arr = avg_lq

    result = np.clip(safe_divide(1.0, avg_lq_arr), 0, 1)

    if is_df:
        return pd.Series(result, index=row_index, name="hachman")
    return result


def specialization_coefficient(
    mat: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Hoover Coefficient of Specialization.

    spec_r = (1/2) * sum_c |s_{rc} - s_c|

    where s_{rc} = share of activity c in region r,
          s_c    = national share of activity c.

    Equivalent to half the Krugman Index.
    Ranges [0, 1]: 0 = region mirrors national structure.

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    total = arr.sum()

    row_sums = arr.sum(axis=1, keepdims=True)
    col_sums = arr.sum(axis=0, keepdims=True)

    s_rc = safe_divide(arr, row_sums)
    s_c = col_sums / total

    result = 0.5 * np.abs(s_rc - s_c).sum(axis=1)

    if is_df:
        return pd.Series(result, index=row_index, name="spec_coeff")
    return result


def krugman_index(
    mat: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Krugman Specialization Index.

    K_r = sum_c |s_{rc} - s_c|

    where s_{rc} = share of activity c in region r,
          s_c    = national share of activity c.

    = 2 * specialization_coefficient.
    Ranges [0, 2].

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    coeff = specialization_coefficient(mat)
    result = 2 * (coeff.values if isinstance(coeff, pd.Series) else coeff)

    if is_df:
        return pd.Series(result, index=row_index, name="krugman_index")
    return result


# Short alias matching the documented API
spec_coefficient = specialization_coefficient
