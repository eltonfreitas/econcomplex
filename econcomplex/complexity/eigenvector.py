"""
Eigenvector method for Economic Complexity (ECI / PCI).

References
----------
Hidalgo & Hausmann (2009); Balland & Rigby (2017).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

from ..core.utils import validate_matrix, safe_divide, normalize_zscore, binarize
from ..core.rca import rca as compute_rca


def _second_eigenvector(mat: np.ndarray) -> np.ndarray:
    """Return the eigenvector corresponding to the second largest eigenvalue.

    The Markov-style co-occurrence matrix (Mcc / Mpp) is in general NOT
    symmetric, because each row is normalised by its own diversity/ubiquity.
    We therefore use the general (non-symmetric) eigensolver ``np.linalg.eig``
    and select the eigenvector associated with the second-largest eigenvalue
    by real part. The largest eigenvalue corresponds to the trivial constant
    vector; the second is the Hidalgo-Hausmann (2009) complexity index.

    Using ``np.linalg.eigh`` here would be incorrect: it assumes a symmetric
    matrix and reads only one triangle, yielding the eigenvector of an
    arbitrarily symmetrised matrix rather than the true second eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    order = np.argsort(eigenvalues.real)
    return np.real(eigenvectors[:, order[-2]])


def eci_pci(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Economic Complexity Index (ECI) and Product Complexity Index (PCI)
    via the eigenvector method.

    Builds Markov matrices:
      Mcc_{rr'} = sum_c (M_{rc}/D_r) * (M_{r'c}/U_c)
      Mpp_{pp'} = sum_r (M_{rp}/U_p) * (M_{rp'}/D_r)

    ECI = second eigenvector of Mcc (sign: positive correlation with diversity).
    PCI = second eigenvector of Mpp (sign: negative correlation with ubiquity).
    Both are z-score normalized.

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
    (eci, pci) as pd.Series or ndarrays.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    kc0 = m.sum(axis=1)  # diversity  R
    kp0 = m.sum(axis=0)  # ubiquity   C

    # Row-normalized and column-normalized matrices
    m_div_kc = safe_divide(m, kc0[:, None])   # M / D_r  (R x C)
    m_div_kp = safe_divide(m, kp0[None, :])   # M / U_c  (R x C)

    # Mcc: R x R
    mcc = m_div_kc @ m_div_kp.T

    # Mpp: C x C
    mpp = m_div_kp.T @ m_div_kc

    # Second eigenvectors
    eci_raw = _second_eigenvector(mcc)
    pci_raw = _second_eigenvector(mpp)

    # Sign correction
    # ECI should correlate positively with diversity
    if np.corrcoef(eci_raw, kc0)[0, 1] < 0:
        eci_raw = -eci_raw
    # PCI should correlate negatively with ubiquity
    if np.corrcoef(pci_raw, kp0)[0, 1] > 0:
        pci_raw = -pci_raw

    eci = normalize_zscore(eci_raw)
    pci = normalize_zscore(pci_raw)

    if is_df:
        return (
            pd.Series(eci, index=row_index, name="eci"),
            pd.Series(pci, index=col_index, name="pci"),
        )
    return eci, pci
