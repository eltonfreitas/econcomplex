"""
Relatedness revealed by flows between activities (e.g. job switches).

References
----------
Neffke, F. & Henning, M. (2013). Skill relatedness and firm
diversification. Strategic Management Journal, 34(3), 297-316.
Neffke, F., Otto, A. & Weyh, A. (2017). Inter-industry labor flows.
Journal of Economic Behavior & Organization, 142, 275-292.
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix


def skill_relatedness(
    flows: Union[np.ndarray, pd.DataFrame],
    symmetric: bool = False,
    exclude_diagonal: bool = True,
    normalize: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Skill relatedness between activities from a matrix of flows.

    Compares the observed flow from activity i to activity j with the flow
    expected from the total outflow of i and the total inflow of j:

    SR_{ij} = F_{ij} / (F_{i.} F_{.j} / F_{..})

    and maps the ratio onto [-1, 1) with (SR - 1) / (SR + 1), so that 0
    means "as many moves as expected", positive values mean related
    activities and -1 means no flow at all.

    The typical input is the number of workers who switch jobs from one
    industry to another, but any origin x destination count works
    (occupations, flows of workers or firms between regions).

    Parameters
    ----------
    flows : array-like (N x N)
        Flow matrix, origins in rows and destinations in columns. A
        DataFrame whose rows and columns do not hold the same labels
        (e.g. the output of ``pivot_to_matrix`` on a long table of
        origin, destination, count) is completed with zeros to the union
        of both.
    symmetric : bool
        If True, average the two directions, giving an undirected
        measure. As in Neffke, Otto & Weyh (2017) the average is taken
        after the transformation. Default False: the measure is directed.
    exclude_diagonal : bool
        If True (default) ignore the diagonal — units that stay in the
        same activity — both in the margins and in the output (NaN).
    normalize : bool
        If True (default) apply (SR - 1) / (SR + 1). If False return the
        raw observed-to-expected ratio (0 to infinity, 1 = as expected).

    Returns
    -------
    N x N matrix (DataFrame if the input was one). Pairs whose expected
    flow is zero (an activity with no outflow or no inflow) are NaN.

    Notes
    -----
    The expected flow from the product of the margins and the
    transformation follow Neffke, Otto & Weyh (2017), who also average
    the transformed values over several years before symmetrizing —
    pool or average the yearly matrices yourself if you have a panel.
    Neffke & Henning (2013) introduced the measure with an expected flow
    predicted by a regression on industry characteristics instead.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> flows = pd.DataFrame([[0, 8, 2], [6, 0, 4], [1, 9, 0]],
    ...                      index=list("ABC"), columns=list("ABC"))
    >>> ec.skill_relatedness(flows).round(2).loc["A"].tolist()
    [nan, 0.17, 0.0]
    """
    is_df = isinstance(flows, pd.DataFrame)
    labels = None
    if is_df:
        labels = flows.index.union(flows.columns, sort=False)
        flows = flows.reindex(index=labels, columns=labels, fill_value=0)

    f = validate_matrix(flows).copy()
    if f.shape[0] != f.shape[1]:
        raise ValueError(f"flows must be square (origin x destination); got {f.shape}.")
    if exclude_diagonal:
        np.fill_diagonal(f, 0.0)

    total = f.sum()
    expected = np.outer(f.sum(axis=1), f.sum(axis=0)) / total if total > 0 else np.zeros_like(f)
    sr = np.full_like(f, np.nan)
    np.divide(f, expected, out=sr, where=expected > 0)

    if normalize:
        sr = (sr - 1) / (sr + 1)
    if symmetric:
        sr = (sr + sr.T) / 2
    if exclude_diagonal:
        np.fill_diagonal(sr, np.nan)

    if is_df:
        return pd.DataFrame(sr, index=labels, columns=labels)
    return sr
