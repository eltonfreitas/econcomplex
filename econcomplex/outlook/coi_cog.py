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

from ..core.utils import (
    validate_matrix, safe_divide, binarize, align_vector, align_square,
)
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

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> pci = pd.Series({"C": 3.0, "A": 1.0, "B": 2.0})   # any order: matched by label
    >>> ec.coi(mat, pci).round(2).tolist()
    [1.67, 1.0, 0.0]
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None

    arr = validate_matrix(mat)
    labels = mat.columns if is_df else None
    pci_arr = align_vector(pci, labels, arr.shape[1], "pci")
    if phi is not None:
        phi = align_square(phi, labels, arr.shape[1])

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
    net: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Complexity Outlook Gain (COG).

    COG_{rc} = (1 - M_{rc}) * sum_{c'} (
        (1 - M_{rc'}) * phi_{cc'} * PCI_{c'} / sum_{c''} phi_{c''c'}
    )

    How much the Complexity Outlook Index of region r would rise if it
    developed activity c: entering c raises the density of every other
    missing activity c' by phi_{cc'} / sum_{c''} phi_{c''c'}, and each of
    those gains is valued at PCI_{c'}. The sum runs over the activities
    the region lacks today, c included, so this is the gross gain: it
    does not subtract the term density_{rc} * PCI_c that leaves the COI
    once c is no longer a missing activity. Cells where the region is
    already specialized are 0.

    This is the form used by the Growth Lab's ``py-ecomplexity``. Up to
    version 1.1.0 the normalization used the proximity total of the
    target activity c instead of that of each c'; the two only coincide
    when every activity has the same total proximity.

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
    net : bool
        If True, subtract density_{rc} * PCI_c, the term of the activity
        itself, so that the result is exactly the change in the
        Complexity Outlook Index when the region enters the activity. It
        can then be negative: entering a complex activity that was the
        region's best nearby option lowers what is left to reach. The
        opportunity gain of the Atlas (Hausmann, Hidalgo et al. 2014) is
        defined net of a term of this kind. Default False, the gross
        gain, as in ``py-ecomplexity``.

    Returns
    -------
    R x C COG matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)
    pci_arr = align_vector(pci, col_index, arr.shape[1], "pci")

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
        phi_arr = align_square(phi, col_index, arr.shape[1])

    # weight of each missing activity c': PCI_{c'} / total proximity of c'
    # (the denominator of its density), spread over the targets c by phi
    weight = safe_divide(pci_arr, phi_arr.sum(axis=0))
    not_present = 1 - m
    cog = not_present * (not_present @ (phi_arr * weight[None, :]).T)
    if net:
        density = safe_divide(m @ phi_arr, phi_arr.sum(axis=0, keepdims=True))
        cog = cog - not_present * density * pci_arr[None, :]

    if is_df:
        return pd.DataFrame(cog, index=row_index, columns=col_index)
    return cog


# Short aliases matching the documented API
coi = complexity_outlook_index
cog = complexity_outlook_gain
