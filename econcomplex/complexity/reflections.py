"""
Method of Reflections (MOR) for Economic Complexity.

References
----------
Hidalgo & Hausmann (2009) "The Building Blocks of Economic Complexity".
Balland & Rigby (2017) for regional adaptation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from ..core.utils import validate_matrix, safe_divide, normalize_zscore, binarize
from ..core.rca import rca as compute_rca


def _binary_presence(mat, use_rca: bool, threshold: float):
    """Binary Mcp plus DataFrame metadata, shared by the MoR variants."""
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None
    arr = validate_matrix(mat)
    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)
    return m, is_df, row_index, col_index


def _reflect(m: np.ndarray, iterations: int, tol: Optional[float] = None):
    """
    Iterate the Method of Reflections from the binary matrix `m`:
      k_{r,n} = (M * k_{c,n-1}) / k_{r,0}
      k_{c,n} = (M^T * k_{r,n-1}) / k_{c,0}
    Optionally stop early when the z-scored reflections change by less
    than `tol`. Returns (kc, kp, kc0, kp0).
    """
    kc0 = m.sum(axis=1)  # diversity
    kp0 = m.sum(axis=0)  # ubiquity
    kc = kc0.copy()
    kp = kp0.copy()
    for _ in range(iterations):
        kc_new = safe_divide(m @ kp, kc0)
        kp_new = safe_divide(m.T @ kc, kp0)
        if tol is not None:
            delta_c = np.max(np.abs(normalize_zscore(kc_new) - normalize_zscore(kc)))
            delta_p = np.max(np.abs(normalize_zscore(kp_new) - normalize_zscore(kp)))
            kc = kc_new
            kp = kp_new
            if delta_c < tol and delta_p < tol:
                break
        else:
            kc = kc_new
            kp = kp_new
    return kc, kp, kc0, kp0


def method_of_reflections(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    iterations: int = 20,
    return_both: bool = True,
    tol: Optional[float] = 1e-10,
) -> Union[Tuple[pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray]]:
    """
    Method of Reflections (iterative) to compute ECI and PCI.

    Starting from the binary Mcp matrix:
      k_{r,0} = diversity (row sums)
      k_{c,0} = ubiquity (col sums)
      k_{r,n} = (M * k_{c,n-1}) / k_{r,0}
      k_{c,n} = (M^T * k_{r,n-1}) / k_{c,0}
    Final values are z-score normalized.

    Parameters
    ----------
    mat : array-like (R x C)
        Value or pre-binarized matrix.
    use_rca : bool
        Compute RCA internally before binarizing.
    threshold : float
        Binarization threshold.
    iterations : int
        Number of reflection steps (default 20).
    return_both : bool
        If True, return (ECI, PCI); if False, return ECI only.
    tol : float, optional
        Convergence tolerance: stop early when the z-scored reflections
        change by less than `tol` between iterations. The check uses the
        z-scored vectors because the raw reflections converge to a
        constant. Pass None to always run all `iterations`.

    Returns
    -------
    (eci, pci) as pd.Series (or ndarrays if input is ndarray).
    """
    m, is_df, row_index, col_index = _binary_presence(mat, use_rca, threshold)
    kc, kp, kc0, kp0 = _reflect(m, iterations, tol=tol)

    eci = normalize_zscore(kc)
    pci = normalize_zscore(kp)

    # Sign orientation, as in the eigenvector method: ECI correlates
    # positively with diversity, PCI negatively with ubiquity
    if np.std(eci) > 0 and np.std(kc0) > 0 and np.corrcoef(eci, kc0)[0, 1] < 0:
        eci = -eci
    if np.std(pci) > 0 and np.std(kp0) > 0 and np.corrcoef(pci, kp0)[0, 1] > 0:
        pci = -pci

    if is_df:
        return (
            pd.Series(eci, index=row_index, name="eci"),
            pd.Series(pci, index=col_index, name="pci"),
        )
    return eci, pci


def mor_regions(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    steps: int = 20,
) -> Union[pd.Series, np.ndarray]:
    """
    Method of Reflections — region/country side only.

    Returns the region complexity at a given reflection step (0–22),
    rescaled to 0–100 for steps > 1.

    step 0 = raw diversity
    step 1 = avg ubiquity of industries present in each region
    step 2+ = higher-order reflections
    """
    m, is_df, row_index, _ = _binary_presence(mat, use_rca, threshold)
    kc, kp, _, _ = _reflect(m, steps)

    if steps > 1:
        mn, mx = kc.min(), kc.max()
        result = (kc - mn) / (mx - mn) * 100 if mx != mn else np.zeros_like(kc)
    else:
        result = kc

    if is_df:
        return pd.Series(result, index=row_index, name=f"mor_region_step{steps}")
    return result


def mor_activities(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    steps: int = 19,
) -> Union[pd.Series, np.ndarray]:
    """
    Method of Reflections — activity/technology side only.

    step 0 = raw ubiquity
    step 1+ = higher-order reflections; rescaled to 0–100 for steps > 1.
    """
    m, is_df, _, col_index = _binary_presence(mat, use_rca, threshold)
    kc, kp, _, _ = _reflect(m, steps)

    if steps > 1:
        mn, mx = kp.min(), kp.max()
        result = (kp - mn) / (mx - mn) * 100 if mx != mn else np.zeros_like(kp)
    else:
        result = kp

    if is_df:
        return pd.Series(result, index=col_index, name=f"mor_activity_step{steps}")
    return result
