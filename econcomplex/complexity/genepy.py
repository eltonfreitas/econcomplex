"""
GENEPY — GENeralised Economic comPlexitY index.

References
----------
Sciarra, C., Chiarotti, G., Ridolfi, L. & Laio, F. (2020). Reconciling
contrasting views on economic complexity. Nature Communications, 11,
3352. Code: https://doi.org/10.5281/zenodo.3876721
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

from ..core.utils import validate_matrix, binarize
from ..core.rca import rca as compute_rca


def _genepy_index(proximity: np.ndarray) -> np.ndarray:
    """GENEPY from a symmetric proximity matrix (diagonal set to zero)."""
    p = proximity.copy()
    np.fill_diagonal(p, 0.0)
    values, vectors = np.linalg.eigh(p)
    top = np.argsort(-np.abs(values), kind="stable")[:2]   # as in the authors' code
    lam, e2 = values[top], vectors[:, top] ** 2             # unit-norm eigenvectors
    return (e2 @ lam) ** 2 + 2 * (e2 @ lam ** 2)


def genepy(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    GENEPY index of regions and activities (advanced implementation;
    prefer ``eci_pci(mat, method="genepy")``, which adds the automatic
    trimming of degenerate units).

    The binary matrix M is rescaled into
    W_{rc} = M_{rc} / (k_r k'_c), with k_r the diversity of region r and
    k'_c = sum_r M_{rc} / k_r. The proximity between regions is
    N = W W^T with the diagonal set to zero. With X_1, X_2 its two
    leading eigenvectors (unit norm) and lambda_1, lambda_2 their
    eigenvalues,

    GENEPY_r = (sum_i lambda_i X_{r,i}^2)^2 + 2 sum_i lambda_i^2 X_{r,i}^2

    The first eigenvector is a linearized Fitness and the second is
    related to the ECI, so the index combines the two views of
    complexity in one non-negative number. Activities use W^T W in the
    same way.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    (genepy_regions, genepy_activities) as pd.Series or ndarrays. Higher
    means more complex; values are not standardized.

    Notes
    -----
    Follows the code released by the authors: the "two leading"
    eigenvalues are the two largest in absolute value. Because the
    diagonal is zero the matrix has negative eigenvalues, and on some
    matrices this rule picks a different second eigenvalue than "the two
    largest".

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> g_regions, g_activities = ec.genepy(mat)
    >>> g_regions.round(3).tolist()
    [0.07, 0.461, 0.531]
    """
    is_df = isinstance(mat, pd.DataFrame)
    arr = validate_matrix(mat)
    m = binarize(compute_rca(arr), threshold) if use_rca else binarize(arr, threshold)

    kc = m.sum(axis=1, keepdims=True)
    if (kc == 0).any() or (m.sum(axis=0) == 0).any():
        raise ValueError(
            "GENEPY is undefined for regions with zero diversity or activities "
            "with zero ubiquity; use eci_pci(mat, method='genepy') or trim_core."
        )
    kp = (m / kc).sum(axis=0, keepdims=True)
    w = m / (kc * kp)
    regions, activities = _genepy_index(w @ w.T), _genepy_index(w.T @ w)

    if is_df:
        return (pd.Series(regions, index=mat.index, name="genepy"),
                pd.Series(activities, index=mat.columns, name="genepy"))
    return regions, activities
