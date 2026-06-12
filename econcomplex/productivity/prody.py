"""
Productivity levels of products and regions.

References
----------
Hausmann, Hwang & Rodrik (2007) "What you export matters".
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, safe_divide
from ..core.rca import rca as compute_rca


def prody(
    mat: Union[np.ndarray, pd.DataFrame],
    gdp: Union[np.ndarray, pd.Series],
    use_rca: bool = True,
) -> Union[pd.Series, np.ndarray]:
    """
    PRODY: Income / productivity level of each activity.

    PRODY_c = sum_r (LQ_{rc} * GDP_r) / sum_r LQ_{rc}

    Weighted average of regional GDP, weighted by each region's
    Revealed Comparative Advantage in activity c.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    gdp : array-like (length R)
        GDP (or income) per region.
    use_rca : bool
        Compute RCA internally (True) or use mat as LQ (False).

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    gdp_arr = np.array(gdp, dtype=float)

    if len(gdp_arr) != arr.shape[0]:
        raise ValueError("Length of gdp must equal number of rows in mat.")

    if use_rca:
        lq = compute_rca(arr)
        if isinstance(lq, pd.DataFrame):
            lq = lq.values
    else:
        lq = arr

    numerator = (lq * gdp_arr[:, None]).sum(axis=0)   # C
    denominator = lq.sum(axis=0)                        # C
    result = safe_divide(numerator, denominator)

    if is_df:
        return pd.Series(result, index=col_index, name="prody")
    return result


def expy(
    mat: Union[np.ndarray, pd.DataFrame],
    gdp: Union[np.ndarray, pd.Series],
    use_rca: bool = True,
) -> Union[pd.Series, np.ndarray]:
    """
    EXPY: Income level of a region's export / activity portfolio.

    EXPY_r = sum_c (s_{rc} * PRODY_c)
    where s_{rc} = x_{rc} / X_r  (share of activity c in region r)

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    gdp : array-like (length R)
        GDP (or income) per region.
    use_rca : bool
        Whether to compute RCA for PRODY calculation.

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    gdp_arr = np.array(gdp, dtype=float)

    prody_arr = prody(arr, gdp_arr, use_rca=use_rca)
    if isinstance(prody_arr, pd.Series):
        prody_arr = prody_arr.values

    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)

    result = (shares * prody_arr[None, :]).sum(axis=1)

    if is_df:
        return pd.Series(result, index=row_index, name="expy")
    return result


def product_gini_index(
    mat: Union[np.ndarray, pd.DataFrame],
    gini_vec: Union[np.ndarray, pd.Series],
    threshold: float = 1.0,
) -> Union[pd.Series, np.ndarray]:
    """
    Product Gini Index (PGI): inequality embedded in a product.

    PGI_c = sum_r (M_{rc} * s_{rc} * Gini_r) / sum_r (M_{rc} * s_{rc})

    Weighted average of regional Gini coefficients, weighted by
    specialization (RCA) shares.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    gini_vec : array-like (length R)
        Gini coefficient per region.
    threshold : float
        RCA binarization threshold.

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    gini_arr = np.array(gini_vec, dtype=float)

    if len(gini_arr) != arr.shape[0]:
        raise ValueError("Length of gini_vec must equal number of rows in mat.")

    rca_mat = compute_rca(arr)
    if isinstance(rca_mat, pd.DataFrame):
        rca_mat = rca_mat.values

    m = (rca_mat >= threshold).astype(float)
    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)

    weights = m * shares  # R x C
    numerator = (weights * gini_arr[:, None]).sum(axis=0)
    denominator = weights.sum(axis=0)
    result = safe_divide(numerator, denominator)

    if is_df:
        return pd.Series(result, index=col_index, name="pgi")
    return result


def product_emissions_index(
    mat: Union[np.ndarray, pd.DataFrame],
    emissions: Union[np.ndarray, pd.Series],
    threshold: float = 1.0,
) -> Union[pd.Series, np.ndarray]:
    """
    Product Emissions Intensity Index (PEII).

    Same formula as PGI but substitutes emissions intensity for Gini:
    PEII_c = sum_r (M_{rc} * s_{rc} * emissions_r) / sum_r (M_{rc} * s_{rc})

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    emissions : array-like (length R)
        Emissions intensity per region.
    threshold : float
        RCA binarization threshold.

    Returns
    -------
    pd.Series indexed by activity.
    """
    is_df = isinstance(mat, pd.DataFrame)
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    em_arr = np.array(emissions, dtype=float)

    if len(em_arr) != arr.shape[0]:
        raise ValueError("Length of emissions must equal number of rows in mat.")

    rca_mat = compute_rca(arr)
    if isinstance(rca_mat, pd.DataFrame):
        rca_mat = rca_mat.values

    m = (rca_mat >= threshold).astype(float)
    row_sums = arr.sum(axis=1, keepdims=True)
    shares = safe_divide(arr, row_sums)

    weights = m * shares
    numerator = (weights * em_arr[:, None]).sum(axis=0)
    denominator = weights.sum(axis=0)
    result = safe_divide(numerator, denominator)

    if is_df:
        return pd.Series(result, index=col_index, name="peii")
    return result


# Short aliases matching the documented API
pgi = product_gini_index
peii = product_emissions_index
