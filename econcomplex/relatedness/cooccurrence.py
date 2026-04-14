"""
Co-occurrence matrices and statistical novelty (z-score).

References
----------
Steijn (2017) probability index; Fleming & Sorenson (2001) for patents.
"""

import numpy as np
import pandas as pd
from typing import Literal, Union

from ..core.utils import validate_matrix, safe_divide


def co_occurrence(
    mat: Union[np.ndarray, pd.DataFrame],
    diagonal: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Co-occurrence matrix: number of times two activities appear together
    across regions/events.

    Cooc = M * M^T  (if mat is events × activities)
         = M^T * M  (if mat is regions × activities — activity co-occurrence)

    Here we compute the *activity-by-activity* co-occurrence:
      Cooc_{cc'} = sum_r M_{rc} * M_{rc'}

    Parameters
    ----------
    mat : array-like (R x C)
        Binary (or value) incidence matrix.
    diagonal : bool
        If False (default), set diagonal to 0.

    Returns
    -------
    C x C co-occurrence matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    cooc = arr.T @ arr

    if not diagonal:
        np.fill_diagonal(cooc, 0.0)

    if is_df:
        return pd.DataFrame(cooc, index=col_index, columns=col_index)
    return cooc


def relatedness_index(
    mat: Union[np.ndarray, pd.DataFrame],
    method: Literal["probability", "association", "cosine", "jaccard"] = "cosine",
    diagonal: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Pairwise relatedness index between activities from co-occurrence.

    Methods
    -------
    "probability" (Steijn 2017):
      SM_{ij} = C_{ij} / (((S_i/T)(S_j/(T-S_i)) + (S_j/T)(S_i/(T-S_j))) * T/2)

    "association" (association strength):
      SA_{ij} = (C_{ij}/T) / ((S_i/T)(S_j/T))

    "cosine":
      SC_{ij} = C_{ij} / sqrt(S_i * S_j)

    "jaccard":
      SJ_{ij} = C_{ij} / (S_i + S_j - C_{ij})

    Parameters
    ----------
    mat : array-like (R x C)
        Binary incidence matrix.
    method : str
        Normalization method.

    Returns
    -------
    C x C normalized relatedness matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    cooc = arr.T @ arr  # C x C
    n_events = arr.shape[0]  # T = number of regions/events
    s = cooc.diagonal().copy()  # S_i = total occurrences of activity i

    if method == "cosine":
        denom = np.sqrt(s[:, None] * s[None, :])
        result = safe_divide(cooc, denom)

    elif method == "jaccard":
        denom = s[:, None] + s[None, :] - cooc
        result = safe_divide(cooc, denom)

    elif method == "association":
        p_i = s / n_events
        expected = p_i[:, None] * p_i[None, :]
        result = safe_divide(cooc / n_events, expected)

    elif method == "probability":
        T = n_events
        si = s.astype(float)
        # expected = ((si/T)(sj/(T-si)) + (sj/T)(si/(T-sj))) * T/2
        t1 = safe_divide(si[:, None] * si[None, :], T * (T - si[:, None]))
        t2 = safe_divide(si[None, :] * si[:, None], T * (T - si[None, :]))
        expected = (t1 + t2) * T / 2
        result = safe_divide(cooc, expected)

    else:
        raise ValueError("method must be 'probability', 'association', 'cosine', or 'jaccard'.")

    if not diagonal:
        np.fill_diagonal(result, 0.0)

    if is_df:
        return pd.DataFrame(result, index=col_index, columns=col_index)
    return result


def z_score_novelty(
    incidence: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Z-score of technological novelty (atypicality of co-occurrence).

    Measures how much each pair of technologies co-occurs more or less
    than expected by chance.

      z_{ij} = (C_{ij} - mu_{ij}) / sigma_{ij}
    where
      mu_{ij}     = n_i * n_j / P
      sigma_{ij}  = sqrt(mu_{ij} * (1 - n_i/P) * (P - n_j)/(P-1))

    Parameters
    ----------
    incidence : array-like (P x T)
        Patent × technology (or event × activity) incidence matrix.

    Returns
    -------
    T x T z-score matrix.
    """
    is_df = isinstance(incidence, pd.DataFrame)
    col_index = incidence.columns if is_df else None

    arr = validate_matrix(incidence)
    P = arr.shape[0]  # number of patents/events

    n = arr.sum(axis=0)  # occurrences per technology (T,)
    cooc = arr.T @ arr   # T x T

    mu = (n[:, None] * n[None, :]) / P  # expected co-occurrence

    variance = mu * (1 - n[:, None] / P) * safe_divide(
        P - n[None, :], P - 1
    )
    sigma = np.sqrt(np.maximum(variance, 0))

    z = safe_divide(cooc - mu, sigma)
    np.fill_diagonal(z, 0.0)

    if is_df:
        return pd.DataFrame(z, index=col_index, columns=col_index)
    return z
