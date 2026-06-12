"""
Relatedness density, distance, and related indicators.

References
----------
Hidalgo et al. (2007); Hausmann & Klinger (2007);
Balland et al. (2019).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide, binarize
from ..core.rca import rca as compute_rca


def _get_mcp_and_phi(
    mat,
    phi,
    use_rca: bool,
    threshold: float,
    proximity_method: str,
):
    """Helper: returns binary M and proximity matrix as ndarrays."""
    is_df = isinstance(mat, pd.DataFrame)

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    if phi is None:
        from .proximity import proximity as _prox
        phi_dict = _prox(
            m, use_rca=False, threshold=0.5,
            method=proximity_method, compute="product",
        )
        phi_arr = (
            phi_dict["product"].values
            if isinstance(phi_dict["product"], pd.DataFrame)
            else phi_dict["product"]
        )
    else:
        phi_arr = phi.values if isinstance(phi, pd.DataFrame) else np.array(phi, dtype=float)

    return m, phi_arr, is_df


def relatedness_density(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relatedness density for each (region, activity) pair.

    density_{rc} = (M * Phi)_{rc} / rowSums(Phi)_c * 100

    Fraction of activities related to c that region r already has,
    expressed as a percentage.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    phi : array-like (C x C), optional
        Pre-computed product proximity matrix. Computed internally if None.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    proximity_method : str
        Normalization method for proximity ('max', 'sqrt', 'min').

    Returns
    -------
    R x C relatedness density matrix (values 0–100).
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)  # 1 x C
    numerator = m @ phi_arr                              # R x C
    density = safe_divide(numerator, col_sums_phi) * 100

    if is_df:
        return pd.DataFrame(density, index=row_index, columns=col_index)
    return density


def distance(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Distance (1 - density/100).

    Weighted fraction of related activities that region r does NOT have.

    Returns
    -------
    R x C distance matrix (values 0–1).
    """
    dens = relatedness_density(mat, phi=phi, use_rca=use_rca,
                                threshold=threshold, proximity_method=proximity_method)
    if isinstance(dens, pd.DataFrame):
        return 1 - dens / 100
    return 1 - dens / 100


def relatedness_density_internal(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Internal relatedness density: density values for activities
    the region ALREADY has (M_{rc} = 1).
    Other cells are NaN.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * 100

    mask = m == 0
    density_internal = density.copy()
    density_internal[mask] = np.nan

    if is_df:
        return pd.DataFrame(density_internal, index=row_index, columns=col_index)
    return density_internal


def relatedness_density_external(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    External relatedness density: density values for activities
    the region does NOT yet have (M_{rc} = 0).
    Other cells are NaN.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * 100

    mask = m == 1
    density_external = density.copy()
    density_external[mask] = np.nan

    if is_df:
        return pd.DataFrame(density_external, index=row_index, columns=col_index)
    return density_external


def relative_relatedness(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relative relatedness (Pinheiro et al. 2021, eq. 7): z-transform of the
    relatedness density against the statistics of the region's option set
    (activities it does NOT currently hold, M_{rc} = 0).

    relative_density_{rc} = (density_{rc} - mean_non_held_r) / std_non_held_r
    for cells where M_{rc} = 0; NaN otherwise.

    References
    ----------
    Pinheiro, Hartmann, Boschma & Hidalgo (2022) "The time and frequency
    of unrelated diversification", Research Policy 51, 104323.

    Returns
    -------
    R x C standardized density matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * 100

    result = np.full_like(density, np.nan)
    for i in range(m.shape[0]):
        non_held = m[i] == 0
        vals = density[i, non_held]
        std = vals.std()
        if std > 0:
            result[i, non_held] = (vals - vals.mean()) / std

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=col_index)
    return result


# Short alias matching the documented API
density = relatedness_density
