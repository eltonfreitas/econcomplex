"""
Subnational / external ECI using externally supplied PCI.

References
----------
Balland & Rigby (2017); Mealy et al. (2019).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

from ..core.utils import validate_matrix, binarize, align_vector
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

    A labelled `pci_external` is matched to the columns of `mat` by label,
    so it may come in any order and carry extra activities; activities of
    `mat` without an external PCI are left out of the mean. Regions with
    no specialization get NaN.

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

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> pci = pd.Series({"C": 3.0, "A": 1.0, "B": 2.0})   # e.g. a national PCI
    >>> eci, _ = ec.subnational_eci(mat, pci, standardize=False)
    >>> eci.tolist()
    [1.0, 2.5, 2.0]
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    pci_arr = align_vector(pci_external, col_index, m.shape[1],
                           "pci_external", allow_missing=True)

    # Activities without an external PCI do not enter the mean
    has_pci = np.isfinite(pci_arr)
    m_valid = m[:, has_pci]
    row_sums = m_valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eci_raw = np.where(row_sums > 0,
                           (m_valid @ pci_arr[has_pci]) / row_sums, np.nan)

    if standardize:
        sd = np.nanstd(eci_raw)
        eci_raw = (eci_raw - np.nanmean(eci_raw)) / sd if sd > 0 else eci_raw * 0.0

    if is_df:
        pci_out = (
            pci_external
            if isinstance(pci_external, pd.Series)
            else pd.Series(pci_arr, index=col_index, name="pci")
        )
        return pd.Series(eci_raw, index=row_index, name="eci_external"), pci_out
    return eci_raw, pci_arr
