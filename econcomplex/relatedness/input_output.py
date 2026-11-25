"""
Relatedness from input-output linkages.

References
----------
Fan, J. P. H. & Lang, L. H. P. (2000). The measurement of relatedness:
an application to corporate diversification. Journal of Business, 73(4),
629-660.
Essletzbichler, J. (2015). Relatedness, industrial branching and
technological cohesion in US metropolitan areas. Regional Studies,
49(5), 752-766.
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, safe_divide


def io_relatedness(
    Z: Union[np.ndarray, pd.DataFrame],
    shares: str = "sales",
    method: str = "mean",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relatedness between industries from an input-output table.

    Two industries are related when a large part of what one sells goes
    to the other, or of what one buys comes from the other. With Z_{ij}
    the value sold by industry i to industry j, the directed strength of
    the link from i to j is

    ``shares="sales"``      a_{ij} = Z_{ij} / sum_j Z_{ij}
                            (share of i's intermediate sales going to j)
    ``shares="purchases"``  b_{ij} = Z_{ij} / sum_i Z_{ij}
                            (share of j's intermediate purchases coming from i)
    ``shares="both"``       (a_{ij} + b_{ij}) / 2

    and the two directions are combined into a symmetric measure by
    their mean (``method="mean"``) or their maximum (``method="max"``).

    Parameters
    ----------
    Z : array-like (N x N)
        Inter-industry flows, sellers in rows and buyers in columns
        (intermediate-use block of an input-output table). A DataFrame
        must have the same labels on both axes.
    shares : {'sales', 'purchases', 'both'}
    method : {'mean', 'max'}

    Returns
    -------
    N x N symmetric matrix in [0, 1], usable as a proximity matrix
    (``phi``) in the density functions once industries are mapped to the
    activities of the value matrix.

    Notes
    -----
    The default, IOR_{ij} = (Z_{ij} / Z_{i.} + Z_{ji} / Z_{j.}) / 2, is
    eq. (1) of Essletzbichler (2015), who follows Fan & Lang (2000) and
    reports it multiplied by 100. The other options are common variants;
    state which one you use.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> Z = pd.DataFrame([[0, 6, 2], [3, 0, 1], [0, 0, 5]],
    ...                  index=list("ABC"), columns=list("ABC"))
    >>> ec.io_relatedness(Z).loc["A"].tolist()     # (6/8 + 3/4) / 2 ; (2/8 + 0/5) / 2
    [0.0, 0.75, 0.125]
    """
    if shares not in ("sales", "purchases", "both"):
        raise ValueError("shares must be 'sales', 'purchases' or 'both'.")
    if method not in ("mean", "max"):
        raise ValueError("method must be 'mean' or 'max'.")
    is_df = isinstance(Z, pd.DataFrame)
    if is_df and not Z.index.equals(Z.columns):
        if set(Z.index) != set(Z.columns):
            raise ValueError("Z must have the same industries in rows and columns.")
        Z = Z.loc[:, Z.index]
    z = validate_matrix(Z)
    if z.shape[0] != z.shape[1]:
        raise ValueError(f"Z must be square (sellers x buyers); got {z.shape}.")

    sales = safe_divide(z, z.sum(axis=1, keepdims=True))
    purchases = safe_divide(z, z.sum(axis=0, keepdims=True))
    link = {"sales": sales, "purchases": purchases, "both": (sales + purchases) / 2}[shares]
    out = (link + link.T) / 2 if method == "mean" else np.maximum(link, link.T)

    if is_df:
        return pd.DataFrame(out, index=Z.index, columns=Z.index)
    return out
