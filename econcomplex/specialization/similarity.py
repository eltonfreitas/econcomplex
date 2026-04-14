"""
Export / portfolio similarity between locations.

References
----------
Bahar et al. (2014) "Neighbors and the Evolution of the Comparative Advantage
of Nations".
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix
from ..core.rca import rca as compute_rca


def export_similarity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    epsilon: float = 0.1,
    log: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Export Similarity Index (Bahar et al. 2014).

    Pearson correlation of (log-)RCA vectors between location pairs.

    SCC_{rr'} = corr(log(RCA_r + epsilon), log(RCA_{r'} + epsilon))

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    use_rca : bool
        Compute RCA internally (True) or treat mat as RCA (False).
    epsilon : float
        Small constant added before log to avoid log(0).
    log : bool
        If True, apply log transform before correlation.

    Returns
    -------
    R x R similarity matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        rca_arr = compute_rca(arr)
        if isinstance(rca_arr, pd.DataFrame):
            rca_arr = rca_arr.values
    else:
        rca_arr = arr

    if log:
        transformed = np.log(rca_arr + epsilon)
    else:
        transformed = rca_arr

    result = np.corrcoef(transformed)  # R x R
    np.fill_diagonal(result, 1.0)

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=row_index)
    return result
