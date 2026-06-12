"""
Subnational / external ECI using externally supplied PCI.

References
----------
Balland & Rigby (2017); Mealy et al. (2019).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

from ..core.utils import validate_matrix, binarize, normalize_zscore
from ..core.rca import rca as compute_rca


def subnational_eci(
    mat: Union[np.ndarray, pd.DataFrame],
    pci_external: Union[np.ndarray, pd.Series],
    use_rca: bool = True,
    threshold: float = 1.0,
    standardize: bool = True,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Subnational ECI using an externally supplied PCI vector.

    Each region's ECI is the mean PCI of activities it has RCA in:
      ECI_r = mean(PCI_c for c where M_{rc} = 1)

    Useful when comparing subnational units to a global product space,
    or when the dataset is too small to compute reliable eigenvectors.

    Parameters
    ----------
    mat : array-like (R x C)
        Regional value matrix.
    pci_external : array-like (length C)
        External PCI vector (e.g., global product complexity index).
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    standardize : bool
        If True, z-score normalize the resulting ECI.

    Returns
    -------
    (eci, pci_external) as pd.Series or ndarrays.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    pci_arr = np.array(pci_external, dtype=float)
    if len(pci_arr) != m.shape[1]:
        raise ValueError("Length of pci_external must match number of columns in mat.")

    row_sums = m.sum(axis=1)
    eci_raw = np.where(
        row_sums > 0,
        (m @ pci_arr) / row_sums,
        0.0,
    )

    if standardize:
        eci_raw = normalize_zscore(eci_raw)

    if is_df:
        pci_out = (
            pci_external
            if isinstance(pci_external, pd.Series)
            else pd.Series(pci_arr, index=col_index, name="pci")
        )
        return pd.Series(eci_raw, index=row_index, name="eci_external"), pci_out
    return eci_raw, pci_arr
