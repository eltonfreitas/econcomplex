"""
Pre-processing utilities for sparse location-activity matrices.

References
----------
Hidalgo & Hausmann (2009); OEC Atlas methodology.
"""

import numpy as np
import pandas as pd
from typing import Union

from .utils import validate_matrix, binarize
from .rca import rca as compute_rca


def trim_core(
    mat: Union[np.ndarray, pd.DataFrame],
    dmin: int = 1,
    umin: int = 1,
    use_rca: bool = True,
    threshold: float = 1.0,
    max_iter: int = 100,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Iteratively prune a value matrix to its (dmin, umin)-core.

    Drops locations (rows) with diversity < dmin and activities (columns)
    with ubiquity < umin, where diversity/ubiquity are counted on the
    binary presence matrix (RCA >= threshold by default). Because removing
    a row or column changes the RCA of the remaining cells, the presence
    matrix is recomputed after each pass until the matrix is stable.

    With the defaults (dmin=1, umin=1) only degenerate units are removed:
    locations with no activity above the RCA threshold and activities
    with no location above it. These units cannot be ranked and distort
    (or break) the complexity calculations. For very sparse networks
    (e.g. subnational trade data) the literature computes complexity on
    the well-connected core, typically dmin=2, umin=2
    (Hidalgo & Hausmann 2009; OEC Atlas).

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    dmin : int
        Minimum diversity for a location to be kept (default 1).
    umin : int
        Minimum ubiquity for an activity to be kept (default 1).
    use_rca : bool
        Compute RCA before binarizing (default True).
    threshold : float
        Binarization threshold (default 1.0).
    max_iter : int
        Safety cap on pruning passes.

    Returns
    -------
    The surviving sub-matrix, same type as the input. With a DataFrame
    the original labels are preserved, so dropped units can be recovered
    by reindexing (`result.reindex(mat.index)`).
    """
    is_df = isinstance(mat, pd.DataFrame)
    work = mat.astype(float) if is_df else pd.DataFrame(validate_matrix(mat))

    # All-zero rows/columns first (always degenerate, whatever the core)
    work = work.loc[work.sum(axis=1) > 0, work.sum(axis=0) > 0]

    for _ in range(max_iter):
        if work.shape[0] == 0 or work.shape[1] == 0:
            break
        if use_rca:
            m = binarize(compute_rca(work.values), threshold)
        else:
            m = binarize(work.values, threshold)
        keep_r = m.sum(axis=1) >= dmin
        keep_c = m.sum(axis=0) >= umin
        if keep_r.all() and keep_c.all():
            break
        work = work.loc[work.index[keep_r], work.columns[keep_c]]
        work = work.loc[work.sum(axis=1) > 0, work.sum(axis=0) > 0]

    return work if is_df else work.values
