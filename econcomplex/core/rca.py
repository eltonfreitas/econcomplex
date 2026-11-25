"""
Revealed Comparative Advantage (RCA) / Balassa Index and variants.

References
----------
Balassa (1965); Hausmann et al. (2007); Balland & Rigby (2017).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from .utils import validate_matrix, safe_divide, binarize, align_vector


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

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> ec.rca(mat).values.tolist()
    [[2.0, 0.0, 0.0], [0.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
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


def rsca(
    mat: Union[np.ndarray, pd.DataFrame],
    method: str = "dalum",
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Revealed Symmetric Comparative Advantage.

    The RCA is asymmetric: de-specialization is squeezed between 0 and 1
    while specialization runs from 1 to infinity. The symmetric versions
    map it onto [-1, 1], with 0 as the neutral value, so that strengths
    and weaknesses are measured on the same scale.

    ``method="dalum"`` (Dalum, Laursen & Villumsen 1998)
        RSCA = (RCA - 1) / (RCA + 1)
    ``method="iapadre"`` (Iapadre 2001)
        RSCA = (RCA_rc - RCA_rC) / (RCA_rc + RCA_rC), where RCA_rc
        compares the share of activity c in region r with its share in
        the *other* regions, and RCA_rC does the same for all the other
        activities taken together. It simplifies to
        (a d - b c) / (a d + b c), with a the value of the cell, b the
        rest of the region, c the rest of the activity and d everything
        else. Unlike the Balassa index, it does not depend on the weight
        of the region itself in the total.

    Both are positive exactly where RCA > 1.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    method : {'dalum', 'iapadre'}

    Returns
    -------
    R x C matrix in [-1, 1] (same type as the input). A cell with zero
    value is -1.

    References
    ----------
    Dalum, B., Laursen, K. & Villumsen, G. (1998). Structural change in
    OECD export specialisation patterns: de-specialisation and
    'stickiness'. International Review of Applied Economics, 12(3),
    423-443.
    Iapadre, P. L. (2001). Measuring international specialization.
    International Advances in Economic Research, 7(2), 173-183.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> ec.rsca(mat).round(2).values.tolist()      # RCA = 2, 0 and 1
    [[0.33, -1.0, -1.0], [-1.0, 0.33, 0.33], [0.0, 0.0, 0.0]]
    """
    is_df = isinstance(mat, pd.DataFrame)
    arr = validate_matrix(mat)
    if method == "dalum":
        r = rca(arr)
        out = (r - 1) / (r + 1)
    elif method == "iapadre":
        row = arr.sum(axis=1, keepdims=True)
        col = arr.sum(axis=0, keepdims=True)
        ad = arr * (arr.sum() - row - col + arr)
        bc = (row - arr) * (col - arr)
        out = np.where(arr > 0, 0.0, -1.0)            # value when ad + bc = 0
        np.divide(ad - bc, ad + bc, out=out, where=(ad + bc) > 0)
    else:
        raise ValueError("method must be 'dalum' or 'iapadre'.")

    if is_df:
        return pd.DataFrame(out, index=mat.index, columns=mat.columns)
    return out


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
    pop_arr = align_vector(pop, index, arr.shape[0], "pop").reshape(-1, 1)  # R × 1

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
