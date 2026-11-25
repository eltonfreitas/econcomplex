"""
Combining indicators into a ranking of diversification options.

Three steps of a "smart specialization" exercise: (i) a desirability
index that combines several indicators of what is worth having, with
explicit weights; (ii) a check of how much the ranking depends on those
weights; (iii) the feasibility x desirability quadrants.

References
----------
Balland, P.-A., Boschma, R., Crespo, J. & Rigby, D. L. (2019). Smart
specialization policy in the European Union: relatedness, knowledge
complexity and regional diversification. Regional Studies, 53(9),
1252-1268.
Pinheiro, F. L., Hartmann, D., Boschma, R. & Hidalgo, C. A. (2022). The
time and frequency of unrelated diversification. Research Policy, 51(8),
104323.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Optional, Union

from ..core.utils import align_vector

QUADRANTS = ("short jump", "long bet", "low value", "avoid")


def _standardized(indicators, standardize) -> pd.DataFrame:
    ind = pd.DataFrame(indicators).astype(float)
    if standardize in (None, False, "none"):
        return ind
    if standardize == "z":
        sd = ind.std(ddof=0)
        out = (ind - ind.mean()) / sd.where(sd > 0)
    elif standardize == "minmax":
        span = ind.max() - ind.min()
        out = (ind - ind.min()) / span.where(span > 0)
    elif standardize == "rank":
        return ind.rank(pct=True)
    else:
        raise ValueError("standardize must be 'z', 'minmax', 'rank' or None.")
    # a constant indicator carries no information: it adds 0 to every score
    return out.where(~(out.isna() & ind.notna()), 0.0)


def _signed_weights(weights, columns) -> np.ndarray:
    if weights is None:
        return np.ones(len(columns))
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    w = align_vector(weights, columns, len(columns), "weights")
    if not np.isfinite(w).all() or np.abs(w).sum() == 0:
        raise ValueError("weights must be finite and not all zero.")
    return w


def desirability_index(
    indicators: pd.DataFrame,
    weights: Optional[Union[dict, pd.Series, list, np.ndarray]] = None,
    standardize: Optional[str] = "z",
) -> pd.Series:
    """
    Weighted combination of indicators of how desirable each activity is.

    Each indicator is put on a common scale and the index is their
    weighted mean:

    D_c = sum_k w_k z_{ck} / sum_k |w_k|

    Typical indicators are the activity's complexity (PCI), average wage,
    growth, linkages, or its association with inequality (PGI) and
    emissions (PEII). The weights are a normative choice and are meant to
    be stated explicitly; use `weight_sensitivity` to report how much the
    ranking depends on them.

    Parameters
    ----------
    indicators : pd.DataFrame (activities x indicators)
        One column per indicator. A dict of Series also works.
    weights : dict, Series or sequence, optional
        One weight per indicator, matched by column name when labelled.
        Equal weights by default. A negative weight makes the indicator
        count against the activity (e.g. emissions intensity).
    standardize : {'z', 'minmax', 'rank', None}
        Common scale applied to each column: z-score (default), min-max
        to [0, 1], percentile rank, or None to use the values as given.

    Returns
    -------
    pd.Series indexed by activity. An activity with a missing indicator
    gets NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> ind = pd.DataFrame({"pci": [1.0, 0.0, -1.0], "wage": [10.0, 30.0, 20.0]},
    ...                    index=["A", "B", "C"])
    >>> ec.desirability_index(ind, weights={"pci": 3, "wage": 1}).round(2).tolist()
    [0.61, 0.31, -0.92]
    """
    z = _standardized(indicators, standardize)
    w = _signed_weights(weights, z.columns)
    score = (z * w).sum(axis=1, skipna=False) / np.abs(w).sum()
    return score.rename("desirability")


def weight_sensitivity(
    indicators: pd.DataFrame,
    weights: Optional[Union[dict, pd.Series, list, np.ndarray]] = None,
    standardize: Optional[str] = "z",
    n_draws: int = 1000,
    concentration: float = 1.0,
    top: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stability of the desirability ranking when the weights change.

    Draws `n_draws` weight vectors from a Dirichlet distribution centred
    on the chosen weights, recomputes `desirability_index` for each draw
    and summarizes where every activity lands. An activity that stays at
    the top whatever the weights is a robust priority; one whose rank
    swings is a priority only under a specific view of what matters.

    Parameters
    ----------
    indicators, weights, standardize
        As in `desirability_index`. The sign of each weight is kept in
        every draw; only the relative magnitudes vary.
    n_draws : int
        Number of weight vectors drawn.
    concentration : float
        Dirichlet concentration. With equal weights, 1 (default) draws
        uniformly over all possible weightings; larger values keep the
        draws closer to the chosen weights.
    top : int
        Size of the "top" group used in `top_share`.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame indexed by activity, sorted by the baseline rank, with:
    ``score`` and ``rank`` (baseline weights; rank 1 is the most
    desirable), ``rank_mean``, ``rank_p05`` and ``rank_p95`` (across the
    draws) and ``top_share`` (share of draws in which the activity is
    among the `top` best). Activities with a missing indicator are left
    out.
    """
    z = _standardized(indicators, standardize).dropna()
    w = _signed_weights(weights, z.columns)
    keep = w != 0
    share = np.abs(w[keep]) / np.abs(w[keep]).sum()

    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(concentration * keep.sum() * share, size=n_draws) * np.sign(w[keep])
    scores = z.to_numpy()[:, keep] @ draws.T                      # activities x draws
    ranks = rankdata(-scores, axis=0, method="min")

    base = z.to_numpy() @ w / np.abs(w).sum()
    out = pd.DataFrame({
        "score": base,
        "rank": rankdata(-base, method="min").astype(int),
        "rank_mean": ranks.mean(axis=1),
        "rank_p05": np.percentile(ranks, 5, axis=1),
        "rank_p95": np.percentile(ranks, 95, axis=1),
        "top_share": (ranks <= top).mean(axis=1),
    }, index=z.index)
    return out.sort_values("rank", kind="stable")


def _options(feasibility, desirability):
    """Feasibility and desirability as R x C arrays, NaN outside the options."""
    is_df = isinstance(feasibility, pd.DataFrame)
    f = np.array(feasibility, dtype=float)
    if f.ndim != 2:
        raise ValueError("feasibility must be a 2-D matrix (regions x activities).")

    if isinstance(desirability, pd.DataFrame) and is_df:
        d = desirability.reindex(index=feasibility.index,
                                 columns=feasibility.columns).to_numpy(dtype=float)
    elif np.ndim(desirability) == 2:
        d = np.asarray(desirability, dtype=float)
        if d.shape != f.shape:
            raise ValueError(f"desirability must be {f.shape}; got {d.shape}.")
    else:
        d = align_vector(desirability, feasibility.columns if is_df else None,
                         f.shape[1], "desirability", allow_missing=True)
        d = np.broadcast_to(d, f.shape)

    valid = ~np.isnan(f) & ~np.isnan(d)
    f, d = np.where(valid, f, np.nan), np.where(valid, d, np.nan)
    return f, d, valid, is_df


def strategy_quadrants(
    feasibility: Union[np.ndarray, pd.DataFrame],
    desirability: Union[np.ndarray, pd.Series, pd.DataFrame],
    by_region: bool = True,
    cut: Union[str, tuple] = "median",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Classify each region's options by feasibility and desirability.

    ==============  ===============  ==========================
    quadrant        feasibility      desirability
    ==============  ===============  ==========================
    ``short jump``  above the cut    above the cut
    ``long bet``    below            above
    ``low value``   above            below
    ``avoid``       below            below
    ==============  ===============  ==========================

    Parameters
    ----------
    feasibility : array-like (R x C)
        How reachable each activity is for each region: relatedness
        density, relative relatedness, a predicted entry probability...
        NaN marks cells that are not options (pass
        ``relatedness_density_external(mat)`` to classify only the
        activities each region does not have).
    desirability : array-like (length C) or (R x C)
        How valuable each activity is: PCI, `desirability_index`, or a
        region-specific measure such as the complexity outlook gain. A
        labelled Series/DataFrame is matched to `feasibility` by label.
    by_region : bool
        If True (default) the cuts are computed within each region, over
        its own options: an option is "feasible" relative to the others
        the same region has, as in relative relatedness. If False a
        single cut is computed over all cells.
    cut : {'median', 'mean'} or (float, float)
        Rule for the cuts, or a fixed pair (feasibility cut,
        desirability cut). "Above" is strict.

    Returns
    -------
    R x C matrix of quadrant names; NaN where the cell is not an option.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> feas = pd.DataFrame([[10.0, 40.0, 30.0, 20.0]], index=["R1"], columns=list("ABCD"))
    >>> pci = pd.Series({"A": 2.0, "B": 1.5, "C": -1.0, "D": -2.0})
    >>> ec.strategy_quadrants(feas, pci).loc["R1"].tolist()
    ['long bet', 'short jump', 'low value', 'avoid']
    """
    f, d, valid, is_df = _options(feasibility, desirability)

    if isinstance(cut, str):
        if cut not in ("median", "mean"):
            raise ValueError("cut must be 'median', 'mean' or a (feasibility, desirability) pair.")
        stat = np.nanmedian if cut == "median" else np.nanmean
        kw = {"axis": 1, "keepdims": True} if by_region else {}
        with warnings.catch_warnings():          # regions without options
            warnings.simplefilter("ignore", RuntimeWarning)
            f_cut, d_cut = stat(f, **kw), stat(d, **kw)
    else:
        f_cut, d_cut = cut

    out = np.full(f.shape, np.nan, dtype=object)
    with np.errstate(invalid="ignore"):
        hi_f, hi_d = f > f_cut, d > d_cut
    for name, sel in zip(QUADRANTS, (hi_f & hi_d, ~hi_f & hi_d, hi_f & ~hi_d, ~hi_f & ~hi_d)):
        out[sel & valid] = name

    if is_df:
        return pd.DataFrame(out, index=feasibility.index, columns=feasibility.columns)
    return out


def opportunity_alignment(
    feasibility: Union[np.ndarray, pd.DataFrame],
    desirability: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = "spearman",
    min_options: int = 3,
) -> Union[pd.Series, np.ndarray]:
    """
    Are a region's nearby opportunities also its valuable ones?

    Correlation, across the options of each region, between feasibility
    and desirability. A positive value means the activities within reach
    are also the desirable ones, so related diversification leads
    somewhere; a negative value means the nearby opportunities are of
    low value and upgrading requires long bets — the region is in a
    low-complexity trap of its own product space.

    Parameters
    ----------
    feasibility : array-like (R x C)
        Density, relative relatedness or a predicted probability; NaN
        marks cells that are not options.
    desirability : array-like (length C) or (R x C)
        PCI, `desirability_index` or a region-specific measure, matched
        by label.
    method : {'spearman', 'pearson'}
        Rank correlation (default) or linear correlation.
    min_options : int
        Regions with fewer options than this get NaN.

    Returns
    -------
    pd.Series indexed by region (array if the input was one), in [-1, 1].
    NaN when feasibility or desirability does not vary among the
    region's options.

    Notes
    -----
    This statistic formalizes a comparison that Mealy & Coyle (2022,
    Fig. 5) make visually, plotting the complexity of the industries a
    local area lacks against their distance to it; the paper does not
    define an index. They also warn that part of the pattern is
    mechanical: places with high ECI tend to have complex activities
    nearby by construction, so compare regions of similar complexity.

    References
    ----------
    Mealy, P. & Coyle, D. (2022). To them that hath: economic complexity
    and local industrial strategy in the UK. International Tax and
    Public Finance, 29(2), 358-377.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'.")
    f, d, valid, is_df = _options(feasibility, desirability)
    out = np.full(f.shape[0], np.nan)
    for r in range(f.shape[0]):
        x, y = f[r, valid[r]], d[r, valid[r]]
        if x.size < min_options:
            continue
        if method == "spearman":
            x, y = rankdata(x), rankdata(y)
        if x.std() > 0 and y.std() > 0:
            out[r] = np.corrcoef(x, y)[0, 1]
    if is_df:
        return pd.Series(out, index=feasibility.index, name="alignment")
    return out
