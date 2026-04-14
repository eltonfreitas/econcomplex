"""
Revealed Comparative Advantage (RCA) / Balassa Index and variants.

References
----------
Balassa (1965); Hausmann et al. (2007); Balland & Rigby (2017).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from .utils import validate_matrix, safe_divide, binarize


def rca(
    mat: Union[np.ndarray, pd.DataFrame],
    binary: bool = False,
    threshold: float = 1.0,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Revealed Comparative Advantage (Balassa Index).

    RCA_{rc} = (x_{rc} / sum_c x_{rc}) / (sum_r x_{rc} / sum_{rc} x_{rc})

    Parameters
    ----------
    mat : array-like (R x C)
        Region × industry (or country × product) value matrix.
    binary : bool
        If True, return binary matrix thresholded at `threshold`.
    threshold : float
        Binarization threshold (default 1.0).

    Returns
    -------
    Same type as input (DataFrame or ndarray).
    """
    is_df = isinstance(mat, pd.DataFrame)
    index = mat.index if is_df else None
    columns = mat.columns if is_df else None

    arr = validate_matrix(mat)
    total = arr.sum()
    if total == 0:
        raise ValueError("Matrix sum is zero; cannot compute RCA.")

    row_sums = arr.sum(axis=1, keepdims=True)   # R × 1
    col_sums = arr.sum(axis=0, keepdims=True)   # 1 × C

    share_rc = safe_divide(arr, row_sums)         # share of industry c in region r
    share_c = col_sums / total                    # national share of industry c

    result = safe_divide(share_rc, share_c)

    if binary:
        result = binarize(result, threshold)

    if is_df:
        return pd.DataFrame(result, index=index, columns=columns)
    return result


def rpop(
    mat: Union[np.ndarray, pd.DataFrame],
    pop: Union[np.ndarray, pd.Series],
    binary: bool = False,
    threshold: float = 1.0,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Population-normalized RCA (RPOP).

    RPOP_{rc} = (x_{rc} / pop_r) / (sum_r x_{rc} / sum_r pop_r)

    Parameters
    ----------
    mat : array-like (R x C)
        Region × industry value matrix.
    pop : 1-D array-like (length R)
        Population vector for each region.
    binary : bool
        If True, binarize at `threshold`.
    threshold : float
        Binarization threshold (default 1.0).
    """
    is_df = isinstance(mat, pd.DataFrame)
    index = mat.index if is_df else None
    columns = mat.columns if is_df else None

    arr = validate_matrix(mat)
    pop_arr = np.array(pop, dtype=float).reshape(-1, 1)  # R × 1

    if len(pop_arr) != arr.shape[0]:
        raise ValueError("Length of pop must equal number of rows in mat.")

    world_pop = pop_arr.sum()
    col_sums = arr.sum(axis=0, keepdims=True)  # 1 × C

    share_rc = safe_divide(arr, pop_arr)               # x_{rc} / pop_r
    share_c = col_sums / world_pop                     # X_c / world_pop

    result = safe_divide(share_rc, share_c)

    if binary:
        result = binarize(result, threshold)

    if is_df:
        return pd.DataFrame(result, index=index, columns=columns)
    return result


def mcp(
    mat: Union[np.ndarray, pd.DataFrame],
    presence_test: str = "rca",
    rca_threshold: float = 1.0,
    rpop_threshold: float = 1.0,
    pop: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Binary presence matrix (Mcp).

    Parameters
    ----------
    mat : array-like (R x C)
        Raw value matrix.
    presence_test : str
        One of 'rca', 'rpop', 'both' (union of rca and rpop), 'manual'
        (treat mat as already binary).
    rca_threshold : float
        RCA binarization threshold.
    rpop_threshold : float
        RPOP binarization threshold.
    pop : array-like (length R), optional
        Required when presence_test is 'rpop' or 'both'.
    """
    is_df = isinstance(mat, pd.DataFrame)

    if presence_test == "manual":
        result = binarize(mat, 0.5)
    elif presence_test == "rca":
        result = binarize(rca(mat), rca_threshold)
    elif presence_test == "rpop":
        if pop is None:
            raise ValueError("pop is required for presence_test='rpop'.")
        result = binarize(rpop(mat, pop), rpop_threshold)
    elif presence_test == "both":
        if pop is None:
            raise ValueError("pop is required for presence_test='both'.")
        m_rca = binarize(rca(mat), rca_threshold)
        m_rpop = binarize(rpop(mat, pop), rpop_threshold)
        result = ((m_rca + m_rpop) > 0).astype(float)
    else:
        raise ValueError(
            "presence_test must be one of 'rca', 'rpop', 'both', 'manual'."
        )

    if is_df:
        index = mat.index
        columns = mat.columns
        return pd.DataFrame(result, index=index, columns=columns)
    return result
