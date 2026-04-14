"""
Complexity Outlook Index (COI) and Complexity Outlook Gain (COG).

References
----------
Hausmann & Hidalgo (2011) "The Atlas of Economic Complexity".
Hidalgo et al. (2007) "The Product Space Conditions the Development of Nations".
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide, binarize
from ..core.rca import rca as compute_rca
from ..relatedness.density import relatedness_density


def complexity_outlook_index(
    mat: Union[np.ndarray, pd.DataFrame],
    pci: Union[np.ndarray, pd.Series],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.Series, np.ndarray]:
    """
    Complexity Outlook Index (COI).

    COI_r = sum_c (density_{rc} * (1 - M_{rc}) * PCI_c)

    Measures a region's potential to diversify into complex activities
    it does not yet have, weighted by how related those activities are
    to its current portfolio.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    pci : array-like (length C)
        Product / activity complexity index.
    phi : array-like (C x C), optional
        Pre-computed proximity matrix. Computed if None.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    proximity_method : str
        Proximity normalization ('max', 'sqrt', 'min').

    Returns
    -------
    pd.Series indexed by region.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    pci_arr = np.array(pci, dtype=float)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    dens = relatedness_density(m, phi=phi, use_rca=False, threshold=0.5,
                                proximity_method=proximity_method)
    if isinstance(dens, pd.DataFrame):
        dens = dens.values

    # COI: sum over non-present activities
    coi = ((dens / 100) * (1 - m) * pci_arr[None, :]).sum(axis=1)

    if is_df:
        return pd.Series(coi, index=row_index, name="coi")
    return coi


def complexity_outlook_gain(
    mat: Union[np.ndarray, pd.DataFrame],
    pci: Union[np.ndarray, pd.Series],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Complexity Outlook Gain (COG).

    COG_{rc} = (1 - M_{rc}) * sum_{c'} (
        (1 - M_{rc'}) * phi_{cc'} * PCI_{c'} / sum_{c''} phi_{cc''}
    )

    Gain in COI if region r were to develop activity c.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    pci : array-like (length C)
        Activity complexity index.
    phi : array-like (C x C), optional
        Pre-computed proximity matrix.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    proximity_method : str
        Proximity normalization method.

    Returns
    -------
    R x C COG matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    pci_arr = np.array(pci, dtype=float)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    if phi is None:
        from ..relatedness.proximity import proximity as _prox
        phi_dict = _prox(m, use_rca=False, threshold=0.5,
                          method=proximity_method, compute="product")
        phi_arr = (
            phi_dict["product"].values
            if isinstance(phi_dict["product"], pd.DataFrame)
            else phi_dict["product"]
        )
    else:
        phi_arr = phi.values if isinstance(phi, pd.DataFrame) else np.array(phi, dtype=float)

    # phi_norm_{cc'} = phi_{cc'} / sum_{c''} phi_{cc''}
    phi_row_sums = phi_arr.sum(axis=1, keepdims=True)
    phi_norm = safe_divide(phi_arr, phi_row_sums)  # C x C

    # For each region r: weighted PCI of non-present activities
    # inner = sum_{c'} (1 - M_{rc'}) * phi_norm_{cc'} * PCI_{c'}
    # Shape: R x C
    # (1 - m): R x C
    # phi_norm * pci: sum over c' of phi_norm[c, c'] * pci[c'] = (phi_norm @ pci): C
    # But conditioned on (1 - m): need element-wise

    cog = np.zeros_like(m)
    phi_pci = phi_norm * pci_arr[None, :]   # C x C, entry [c, c'] = phi_norm[c,c'] * PCI[c']

    for r in range(m.shape[0]):
        not_present = (1 - m[r])  # C
        # For each target activity c: sum_{c'} (1 - m[r,c']) * phi_norm[c, c'] * PCI[c']
        cog[r] = not_present * (phi_pci @ not_present)

    # Mask to non-present only
    cog = cog * (1 - m)

    if is_df:
        return pd.DataFrame(cog, index=row_index, columns=col_index)
    return cog
