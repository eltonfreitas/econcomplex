"""
Cross-space proximity and relatedness between two different activity spaces.

References
----------
Catalan et al. (2020) "Cross-space proximity and relatedness".
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide, binarize
from ..core.rca import rca as compute_rca


def cross_proximity(
    mat_a: Union[np.ndarray, pd.DataFrame],
    mat_b: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Cross-space proximity between activity space A and activity space B.

    X_phi_{ij} = min(C_{ij}/U_j^B, C_{ij}/U_i^A)
    where C_{ij} = (M_A^T * M_B)_{ij} = co-presence count

    Parameters
    ----------
    mat_a : array-like (R x A)
        Region × activity-A matrix.
    mat_b : array-like (R x B)
        Region × activity-B matrix.
    use_rca : bool
        Compute RCA before binarizing for both matrices.
    threshold : float
        Binarization threshold.

    Returns
    -------
    A x B cross-proximity matrix.
    """
    is_df_a = isinstance(mat_a, pd.DataFrame)
    is_df_b = isinstance(mat_b, pd.DataFrame)
    col_a = mat_a.columns if is_df_a else None
    col_b = mat_b.columns if is_df_b else None

    arr_a = validate_matrix(mat_a)
    arr_b = validate_matrix(mat_b)

    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError("mat_a and mat_b must have the same number of rows (locations).")

    if use_rca:
        m_a = binarize(compute_rca(arr_a), threshold)
        m_b = binarize(compute_rca(arr_b), threshold)
    else:
        m_a = binarize(arr_a, threshold)
        m_b = binarize(arr_b, threshold)

    # Co-presence: A x B
    c = m_a.T @ m_b  # (A x R) @ (R x B) = A x B

    ub_a = m_a.sum(axis=0)  # A
    ub_b = m_b.sum(axis=0)  # B

    # min(C/U_b, C/U_a) = C / max(U_a, U_b)
    denom = np.maximum(ub_a[:, None], ub_b[None, :])
    result = safe_divide(c, denom)

    if is_df_a and is_df_b:
        return pd.DataFrame(result, index=col_a, columns=col_b)
    return result


def cross_relatedness(
    mat_a: Union[np.ndarray, pd.DataFrame],
    x_phi: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Cross-space relatedness density.

    x_density_{ri} = (M_A * X_phi)_{ri} / (ones * X_phi)_{ri}

    Probability that a region holds activity i (space A) given its presence
    in related activities of space B.

    Parameters
    ----------
    mat_a : array-like (R x A)
        Region × activity-A value matrix.
    x_phi : array-like (A x B)
        Cross-proximity matrix from `cross_proximity`.
    use_rca : bool
        Compute RCA before binarizing mat_a.
    threshold : float
        Binarization threshold.

    Returns
    -------
    R x A cross-relatedness matrix.
    """
    is_df = isinstance(mat_a, pd.DataFrame)
    row_index = mat_a.index if is_df else None
    col_a = mat_a.columns if is_df else None

    arr_a = validate_matrix(mat_a)
    phi_arr = (
        x_phi.values if isinstance(x_phi, pd.DataFrame) else np.array(x_phi, dtype=float)
    )

    if use_rca:
        m_a = binarize(compute_rca(arr_a), threshold)
    else:
        m_a = binarize(arr_a, threshold)

    # numerator: R x B
    numerator = m_a @ phi_arr

    # denominator: 1 x B (sum of each column of x_phi)
    denom = phi_arr.sum(axis=0, keepdims=True)

    result = safe_divide(numerator, denom)

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=col_a)
    return result
