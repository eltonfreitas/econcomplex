"""
Product Space proximity matrices.

References
----------
Hidalgo et al. (2007) "The Product Space Conditions the Development of Nations".
"""

import numpy as np
import pandas as pd
from typing import Literal, Union

from ..core.utils import validate_matrix, safe_divide, binarize
from ..core.rca import rca as compute_rca


def proximity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    method: Literal["max", "sqrt", "min"] = "max",
    compute: Literal["product", "location", "both"] = "both",
    continuous: bool = False,
    continuous_method: Literal["correlation", "cosine"] = "correlation",
) -> dict:
    """
    Compute product and/or location proximity matrices.

    Product proximity (phi_pp'):
      numerator    = M^T * M  (co-export / co-presence count)
      "max"   norm = max(U_p, U_p')
      "sqrt"  norm = sqrt(U_p * U_p')   (geometric mean, cosine-like)
      "min"   norm = min(U_p, U_p')     (conditional probability)

    Location proximity (phi_rr'):
      Same structure but on M * M^T, normalized by diversity.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    method : str
        Normalization method: 'max', 'sqrt', or 'min'.
    compute : str
        Which side to compute: 'product', 'location', or 'both'.
    continuous : bool
        If True, skip binarization and compute the proximity on the
        continuous RCA values (see `continuous_proximity`); `method`
        and `threshold` are ignored.
    continuous_method : str
        Similarity used when `continuous=True`: 'correlation'
        (Pearson, rescaled to [0, 1]) or 'cosine'.

    Returns
    -------
    dict with keys 'product' and/or 'location' as DataFrames (or ndarrays).
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if continuous:
        rca_arr = compute_rca(arr) if use_rca else arr
        results = {}
        if compute in ("product", "both"):
            phi_p = continuous_proximity(rca_arr, method=continuous_method)
            if is_df:
                results["product"] = pd.DataFrame(phi_p, index=col_index, columns=col_index)
            else:
                results["product"] = phi_p
        if compute in ("location", "both"):
            phi_l = continuous_proximity(rca_arr.T, method=continuous_method)
            if is_df:
                results["location"] = pd.DataFrame(phi_l, index=row_index, columns=row_index)
            else:
                results["location"] = phi_l
        return results

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    results = {}

    def _normalize(cooc: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Normalize co-occurrence matrix by row/col counts."""
        if method == "max":
            denom = np.maximum(counts[:, None], counts[None, :])
        elif method == "sqrt":
            denom = np.sqrt(counts[:, None] * counts[None, :])
        elif method == "min":
            denom = np.minimum(counts[:, None], counts[None, :])
        else:
            raise ValueError("method must be 'max', 'sqrt', or 'min'.")
        return safe_divide(cooc, denom)

    if compute in ("product", "both"):
        ubiq = m.sum(axis=0)   # C
        cooc_p = m.T @ m       # C x C
        phi_p = _normalize(cooc_p, ubiq)
        np.fill_diagonal(phi_p, 0.0)
        if is_df:
            results["product"] = pd.DataFrame(phi_p, index=col_index, columns=col_index)
        else:
            results["product"] = phi_p

    if compute in ("location", "both"):
        div = m.sum(axis=1)    # R
        cooc_l = m @ m.T       # R x R
        phi_l = _normalize(cooc_l, div)
        np.fill_diagonal(phi_l, 0.0)
        if is_df:
            results["location"] = pd.DataFrame(phi_l, index=row_index, columns=row_index)
        else:
            results["location"] = phi_l

    return results


def continuous_proximity(
    rca_mat: Union[np.ndarray, pd.DataFrame],
    method: Literal["correlation", "cosine"] = "correlation",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Continuous product proximity from a (continuous) RCA matrix.

    method='correlation' (default):
      phi_{pp'} = (1 + corr(RCA_p, RCA_{p'})) / 2
      (Pearson correlation rescaled from [-1, 1] to [0, 1])
    method='cosine':
      phi_{pp'} = (RCA_p . RCA_{p'}) / (||RCA_p|| * ||RCA_{p'}||)

    Parameters
    ----------
    rca_mat : array-like (R x C)
        Pre-computed (continuous) RCA matrix.
    method : str
        'correlation' or 'cosine'.

    Returns
    -------
    C x C proximity matrix with zero diagonal.
    """
    is_df = isinstance(rca_mat, pd.DataFrame)
    col_index = rca_mat.columns if is_df else None

    arr = validate_matrix(rca_mat)

    if method == "correlation":
        corr = np.corrcoef(arr.T)  # C x C
        phi = (1 + corr) / 2.0
    elif method == "cosine":
        norms = np.linalg.norm(arr, axis=0)  # C
        phi = safe_divide(arr.T @ arr, norms[:, None] * norms[None, :])
    else:
        raise ValueError("method must be 'correlation' or 'cosine'.")
    np.fill_diagonal(phi, 0.0)

    if is_df:
        return pd.DataFrame(phi, index=col_index, columns=col_index)
    return phi


def _continuous_on_values(mat, use_rca, method):
    """Shortcut: RCA (optional) + continuous_proximity with given method."""
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None
    arr = validate_matrix(mat)
    rca_arr = compute_rca(arr) if use_rca else arr
    phi = continuous_proximity(rca_arr, method=method)
    if is_df:
        return pd.DataFrame(phi, index=col_index, columns=col_index)
    return phi


def cosine_proximity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Shortcut for `continuous_proximity(rca(mat), method='cosine')`:
    cosine similarity between the RCA vectors of each pair of activities.

    Returns a C x C proximity matrix with zero diagonal.
    """
    return _continuous_on_values(mat, use_rca, "cosine")


def correlation_proximity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Shortcut for `continuous_proximity(rca(mat), method='correlation')`:
    Pearson correlation between RCA vectors, rescaled to [0, 1].

    Returns a C x C proximity matrix with zero diagonal.
    """
    return _continuous_on_values(mat, use_rca, "correlation")


# Documented-API alias: relatedness(mat, phi) == relatedness_density
from .density import relatedness_density as _relatedness_density  # noqa: E402

relatedness = _relatedness_density


