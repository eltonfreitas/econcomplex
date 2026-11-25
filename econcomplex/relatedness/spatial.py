"""
Spatial density: specialization of a region's neighbours.

References
----------
Bahar, D., Hausmann, R. & Hidalgo, C. A. (2014). Neighbors and the
evolution of the comparative advantage of nations. Journal of
International Economics, 92(1), 111-123.
Boschma, R., Martín, V. & Minondo, A. (2017). Neighbour regions as the
source of new industries. Papers in Regional Science, 96(2), 227-246.
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, binarize, align_square
from ..core.rca import rca as compute_rca


def spatial_density(
    mat: Union[np.ndarray, pd.DataFrame],
    W: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Share of a region's neighbours that are specialized in each activity.

    SD_{rc} = sum_{r'} w_{rr'} M_{r'c} / sum_{r'} w_{rr'}

    The spatial counterpart of relatedness density: instead of asking how
    much of what is related to activity c the region already has, it asks
    how much of the region's neighbourhood already has c. The weights are
    supplied by the user, so any notion of neighbourhood works —
    contiguity (0/1), inverse distance, commuting or migration flows —
    and the library needs no geographic dependency.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    W : array-like (R x R)
        Spatial weights, w_{rr'} >= 0 being the weight of region r' as a
        neighbour of r. A DataFrame is matched to the regions of `mat`
        by label. The diagonal is ignored (a region is not its own
        neighbour). W does not need to be symmetric or row-standardized.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    R x C matrix of shares in [0, 1]. Regions without neighbours (zero
    total weight) are NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> W = pd.DataFrame([[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    ...                  index=mat.index, columns=mat.index)
    >>> ec.spatial_density(mat, W).loc["R1"].tolist()   # neighbours: R2 and R3
    [0.5, 1.0, 1.0]
    """
    is_df = isinstance(mat, pd.DataFrame)
    arr = validate_matrix(mat)
    m = binarize(compute_rca(arr), threshold) if use_rca else binarize(arr, threshold)

    w = align_square(W, mat.index if is_df else None, arr.shape[0], "W").copy()
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("W must hold finite, non-negative weights.")
    np.fill_diagonal(w, 0.0)

    total = w.sum(axis=1, keepdims=True)
    sd = np.full_like(m, np.nan, dtype=float)
    np.divide(w @ m, total, out=sd, where=total > 0)

    if is_df:
        return pd.DataFrame(sd, index=mat.index, columns=mat.columns)
    return sd
