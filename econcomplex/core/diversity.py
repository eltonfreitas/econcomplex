"""
Diversity and ubiquity indicators.
"""

import numpy as np
import pandas as pd
from typing import Union

from .utils import validate_matrix, safe_divide
from .rca import rca as compute_rca


def diversity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.Series:
    """
    Diversity: number of activities in which each region has RCA >= threshold.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix or pre-computed binary Mcp matrix.
    use_rca : bool
        If True, compute RCA internally before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    index = mat.index if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        rca_mat = compute_rca(arr)
        m = (rca_mat >= threshold).astype(float)
    else:
        m = (arr >= threshold).astype(float)

    result = m.sum(axis=1)

    if is_df:
        return pd.Series(result, index=index, name="diversity")
    return result


def ubiquity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.Series:
    """
    Ubiquity: number of regions where each activity has RCA >= threshold.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix or pre-computed binary Mcp matrix.
    use_rca : bool
        If True, compute RCA internally before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    columns = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        rca_mat = compute_rca(arr)
        m = (rca_mat >= threshold).astype(float)
    else:
        m = (arr >= threshold).astype(float)

    result = m.sum(axis=0)

    if is_df:
        return pd.Series(result, index=columns, name="ubiquity")
    return result


def normalized_ubiquity(
    mat: Union[np.ndarray, pd.DataFrame],
    threshold: float = 1.0,
) -> pd.Series:
    """
    Normalized ubiquity: ratio of activity's share of total value to
    its share of ubiquity.

    norm_ubiquity_c = (sum_r x_{rc} / sum_{rc} x_{rc}) / (U_c / sum_c U_c)

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    columns = mat.columns if is_df else None

    arr = validate_matrix(mat)
    total = arr.sum()
    ub = ubiquity(arr, use_rca=True, threshold=threshold)

    if isinstance(ub, pd.Series):
        ub_arr = ub.values
    else:
        ub_arr = ub

    col_sums = arr.sum(axis=0)
    value_share = col_sums / total
    ubiquity_share = safe_divide(ub_arr, ub_arr.sum())

    result = safe_divide(value_share, ubiquity_share)

    if is_df:
        return pd.Series(result, index=columns, name="norm_ubiquity")
    return result
