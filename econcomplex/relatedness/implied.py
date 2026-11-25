"""
Implied comparative advantage.

References
----------
Hausmann, R., Stock, D. P. & Yıldırım, M. A. (2022). Implied comparative
advantage. Research Policy, 51(8), 104143.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix
from ..core.rca import rca as compute_rca


def _neighbour_weights(vectors: np.ndarray, k: Optional[int]) -> np.ndarray:
    """
    Row-stochastic weights of the k most similar rows of `vectors`
    (similarity = (1 + Pearson correlation) / 2; the row itself excluded).
    """
    n = vectors.shape[0]
    k = max(1, min(n - 1, int(round(np.sqrt(n))) if k is None else int(k)))
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (1 + np.corrcoef(vectors)) / 2
    sim = np.where(np.isnan(sim), -np.inf, sim)        # constant vectors: never neighbours
    np.fill_diagonal(sim, -np.inf)

    keep = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    w = np.zeros_like(sim)
    rows = np.arange(n)[:, None]
    w[rows, keep] = sim[rows, keep]
    w[~np.isfinite(w)] = 0.0
    total = w.sum(axis=1, keepdims=True)
    return np.divide(w, total, out=np.full_like(w, np.nan), where=total > 0)


def implied_rca(
    mat: Union[np.ndarray, pd.DataFrame],
    space: str = "activity",
    k: Optional[int] = None,
    cap: Optional[float] = 5.0,
    use_rca: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Implied comparative advantage (Hausmann, Stock & Yıldırım 2022).

    Predicts the specialization of region r in activity c from what
    similar units do, using the intensities themselves instead of a
    binary presence:

    ``space="activity"`` (their "product space density")
        weighted mean of the RCA of r in the k activities most similar
        to c;
    ``space="region"`` (their "country space density")
        weighted mean of the RCA in c of the k regions most similar to r;
    ``space="mean"``
        simple average of the two.

    Similarity between two activities is (1 + corr) / 2, with corr the
    Pearson correlation of their RCA vectors across regions (across
    activities for two regions). The weights are the similarities of the
    k nearest neighbours, normalized to add up to 1; the unit itself is
    never its own neighbour, so the implied value of a cell does not use
    that cell directly.

    The gap between the implied and the observed RCA is the paper's
    predictor of growth: a cell far below its implied value tends to
    grow. For entry, pass the implied RCA as a layer to `build_features`
    or as a score to `validate_entry`.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix (or a specialization matrix, with ``use_rca=False``).
    space : {'activity', 'region', 'mean'}
    k : int, optional
        Number of neighbours. Default: the square root of the number of
        activities (or regions), as in the paper.
    cap : float, optional
        Ceiling applied to the RCA when computing similarities only, so
        that a few extreme values do not dominate the correlations
        (5 in the paper). None for no ceiling.
    use_rca : bool
        If True (default) compute the RCA of `mat`. If False use `mat` as
        the specialization measure — e.g. ``rpop(mat, pop)``, the
        per-capita advantage that is the paper's main variable.

    Returns
    -------
    R x C matrix of implied RCA, in the units of the RCA.

    Notes
    -----
    In the paper the two densities enter a regression jointly (in logs);
    the average offered by ``space="mean"`` is the combination used in
    its simulations.

    Examples
    --------
    >>> import econcomplex as ec
    >>> mat = ec.pivot_to_matrix(ec.make_sample_data(30, 15, seed=42), "loc", "act", "val")
    >>> ec.implied_rca(mat).shape
    (30, 15)
    """
    if space not in ("activity", "region", "mean"):
        raise ValueError("space must be 'activity', 'region' or 'mean'.")
    is_df = isinstance(mat, pd.DataFrame)
    arr = validate_matrix(mat)
    r = compute_rca(arr) if use_rca else arr
    capped = r if cap is None else np.minimum(r, cap)

    parts = []
    if space in ("activity", "mean"):
        parts.append(r @ _neighbour_weights(capped.T, k).T)
    if space in ("region", "mean"):
        parts.append(_neighbour_weights(capped, k) @ r)
    implied = sum(parts) / len(parts)

    if is_df:
        return pd.DataFrame(implied, index=mat.index, columns=mat.columns)
    return implied
