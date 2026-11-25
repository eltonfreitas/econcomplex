"""
Predicting the entry of regions into new activities: targets and
out-of-sample validation of any predictor.

The functions here are model-agnostic. `entry_targets` defines what an
entry is and which cells are at risk; `validate_entry` scores any set of
predictors — relatedness density, a model's predicted probability, a
rule of thumb — against the entries that happened.

References
----------
Hidalgo, C. A., Klinger, B., Barabási, A.-L. & Hausmann, R. (2007). The
product space conditions the development of nations. Science,
317(5837), 482-487.
Li, Y. & Neffke, F. (2024). Evaluating the principle of relatedness:
estimation, drivers and implications for policy. Research Policy,
53(3), 104952.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Optional, Sequence, Union

from ..core.utils import validate_matrix, align_matrix


def entry_targets(
    rca_t0: Union[np.ndarray, pd.DataFrame],
    rca_t1: Union[np.ndarray, pd.DataFrame],
    values_t1: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    lower: float = 0.5,
    upper: float = 1.0,
    min_value: Optional[float] = None,
):
    """
    Binary entry target and risk set between two periods.

    A cell (region, activity) is *at risk* when the region is clearly not
    specialized at the start, RCA_t0 < `lower`. It *enters* when it
    reaches RCA_t1 >= `upper` and, optionally, a minimum size. Leaving a
    gap between `lower` and `upper` (0.5 and 1 by default) avoids
    counting as entries the cells that merely hover around RCA = 1.

    Parameters
    ----------
    rca_t0, rca_t1 : array-like (R x C)
        RCA (or any specialization index, e.g. `rpop`) at the start and
        at the end of the window. With DataFrames, `rca_t1` is matched to
        `rca_t0` by label; regions or activities absent at the end count
        as RCA = 0.
    values_t1 : array-like (R x C), optional
        Values at the end of the window (employment, exports), used with
        `min_value`.
    lower : float
        A cell is at risk if RCA_t0 < lower.
    upper : float
        A cell enters if RCA_t1 >= upper.
    min_value : float, optional
        Additionally require values_t1 >= min_value, so that a tiny
        activity in a tiny region does not count as an entry.

    Returns
    -------
    target : R x C matrix
        1.0 where the cell entered, 0.0 where it was at risk and did not,
        NaN where it was not at risk.
    at_risk : R x C boolean matrix

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> r0 = pd.DataFrame([[0.2, 1.4, 0.0]], index=["R1"], columns=list("ABC"))
    >>> r1 = pd.DataFrame([[1.3, 1.1, 0.4]], index=["R1"], columns=list("ABC"))
    >>> target, at_risk = ec.entry_targets(r0, r1)
    >>> target.loc["R1"].tolist()
    [1.0, nan, 0.0]
    """
    if (min_value is None) != (values_t1 is None):
        raise ValueError("values_t1 and min_value must be given together.")
    is_df = isinstance(rca_t0, pd.DataFrame)

    def like_t0(obj, name):
        if is_df and isinstance(obj, pd.DataFrame):
            obj = obj.reindex(index=rca_t0.index, columns=rca_t0.columns, fill_value=0).fillna(0)
        arr = validate_matrix(obj)
        if arr.shape != r0.shape:
            raise ValueError(f"{name} must be {r0.shape}; got {arr.shape}.")
        return arr

    r0 = validate_matrix(rca_t0)
    at_risk = r0 < lower
    entered = like_t0(rca_t1, "rca_t1") >= upper
    if min_value is not None:
        entered &= like_t0(values_t1, "values_t1") >= min_value
    target = np.where(at_risk, entered.astype(float), np.nan)

    if is_df:
        kw = {"index": rca_t0.index, "columns": rca_t0.columns}
        return pd.DataFrame(target, **kw), pd.DataFrame(at_risk, **kw)
    return target, at_risk


def potential_employment(
    prob: Union[np.ndarray, pd.DataFrame],
    values: Union[np.ndarray, pd.DataFrame],
    arrival: str = "lq1",
    entrants: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """
    Expected size of the activities a region may enter.

    Turns entry probabilities into quantities (jobs, exports):

    expected_{rc} = prob_{rc} * arrival_{rc}

    where arrival is the size the activity would have in the region once
    established. Two definitions are offered:

    ``arrival="lq1"``
        The size at which the region is exactly as specialized as the
        aggregate (location quotient of 1):
        arrival_{rc} = E_{r.} E_{.c} / E_{..}.
    ``arrival="entrants"``
        What past entrants actually reached: quantiles, by activity, of
        the sizes observed in `entrants`. One scenario per quantile
        (``q25``, ``q50``, ``q75`` by default).

    Parameters
    ----------
    prob : array-like (R x C)
        Probability of entry (ideally calibrated, see `predict_entry`).
        NaN marks cells that are not options; they are left out.
    values : array-like (R x C)
        Current value matrix (employment, exports).
    arrival : {'lq1', 'entrants'}
    entrants : array-like (R x C), optional
        Sizes reached by observed entries, NaN elsewhere — e.g.
        ``values_t1.where(target == 1)`` from a past window. Required
        for ``arrival="entrants"``.
    quantiles : sequence of float
        Quantiles for ``arrival="entrants"``.

    Returns
    -------
    pd.DataFrame indexed by (region, activity, scenario) with columns
    ``probability``, ``arrival`` and ``expected``. Sum ``expected`` by
    region or by activity for totals; activities without any observed
    entrant have NaN in the ``entrants`` scenarios.

    Notes
    -----
    The sum of expected sizes is an expectation over many independent
    entries, not a forecast for a single cell, and it inherits any
    miscalibration of the probabilities.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> values = pd.DataFrame([[30, 10], [20, 40]], index=["R1", "R2"], columns=["A", "B"])
    >>> prob = pd.DataFrame([[None, 0.5], [None, None]], index=values.index,
    ...                     columns=values.columns, dtype=float)
    >>> ec.potential_employment(prob, values)["expected"].tolist()   # 0.5 * 40 * 50 / 100
    [10.0]
    """
    is_df = isinstance(values, pd.DataFrame)
    v = validate_matrix(values)
    rows = values.index if is_df else pd.RangeIndex(v.shape[0])
    cols = values.columns if is_df else pd.RangeIndex(v.shape[1])
    p = align_matrix(prob, rows, cols, "prob")

    if arrival == "lq1":
        total = v.sum()
        size = np.outer(v.sum(axis=1), v.sum(axis=0)) / total if total > 0 else np.zeros_like(v)
        scenarios = {"lq1": size}
    elif arrival == "entrants":
        if entrants is None:
            raise ValueError("arrival='entrants' needs the `entrants` matrix.")
        e = align_matrix(entrants, rows, cols, "entrants")
        seen = ~np.isnan(e).all(axis=0)
        scenarios = {}
        for q in quantiles:
            by_activity = np.full(v.shape[1], np.nan)
            by_activity[seen] = np.nanquantile(e[:, seen], q, axis=0)
            scenarios[f"q{round(100 * q):02d}"] = np.broadcast_to(by_activity, v.shape)
    else:
        raise ValueError("arrival must be 'lq1' or 'entrants'.")

    r, c = np.nonzero(~np.isnan(p))
    parts = []
    for name, size in scenarios.items():
        parts.append(pd.DataFrame({
            "region": rows[r], "activity": cols[c], "scenario": name,
            "probability": p[r, c], "arrival": size[r, c],
            "expected": p[r, c] * size[r, c],
        }))
    out = pd.concat(parts, ignore_index=True)
    return out.set_index(["region", "activity", "scenario"])


# ── Metrics (numpy/scipy only; same definitions as scikit-learn) ──────────────
def _auc(y, s):
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (rankdata(s)[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(y, s):
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]     # end of each tie block
    tp = np.cumsum(y)[last]
    return float(np.sum(np.diff(np.r_[0, tp / n_pos]) * tp / (last + 1)))


def _region_metrics(Y, S, ks):
    """Per-region AUC and hits among the top k (regions x 1 + len(ks))."""
    R = Y.shape[0]
    auc = np.full(R, np.nan)
    hits = np.full((R, len(ks)), np.nan)
    size = np.zeros(R)
    for r in range(R):
        ok = ~np.isnan(Y[r]) & ~np.isnan(S[r])
        size[r] = ok.sum()
        if not size[r]:
            continue
        y, s = Y[r, ok] > 0.5, S[r, ok]
        auc[r] = _auc(y, s)
        top = y[np.argsort(-s, kind="stable")]
        hits[r] = [top[:k].sum() for k in ks]
    return auc, hits, size


def _nanmean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _metrics(Y, S, ks, per_region, rows=None):
    """All metrics of one score on the regions `rows` (bootstrap sample)."""
    auc_r, hits, size, n_pos = per_region
    if rows is not None:
        Y, S = Y[rows], S[rows]
        auc_r, hits, size, n_pos = auc_r[rows], hits[rows], size[rows], n_pos[rows]
    ok = ~np.isnan(Y) & ~np.isnan(S)
    y, s = Y[ok] > 0.5, S[ok]

    out = {"auc": _auc(y, s) if y.size else np.nan,
           "auc_within": _nanmean(auc_r),
           "average_precision": _average_precision(y, s) if y.size else np.nan}
    has, pos = size > 0, n_pos > 0
    for j, k in enumerate(ks):
        out[f"precision@{k}"] = _nanmean(hits[has, j] / np.minimum(k, size[has]))
        out[f"recall@{k}"] = _nanmean(hits[pos, j] / n_pos[pos])
    probs = y.size and s.min() >= 0 and s.max() <= 1
    out["brier"] = float(np.mean((s - y) ** 2)) if probs else np.nan
    return out


def validate_entry(
    scores: Union[np.ndarray, pd.DataFrame, dict],
    target: Union[np.ndarray, pd.DataFrame],
    k: Sequence[int] = (10, 20),
    baseline: Optional[str] = None,
    n_boot: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    How well do scores computed at t0 predict the entries observed by t1?

    Evaluates one or more predictors of entry against the target of
    `entry_targets`. A predictor is any R x C matrix in which a higher
    value means "more likely to enter": relatedness density, relative
    relatedness, the probability predicted by a model, the initial RCA
    (persistence), a rule of thumb. For an honest test, everything behind
    a score — including the proximity matrix — must use data up to t0
    only, and model predictions must be out of sample (trained on other
    regions or on an earlier window).

    Metrics
    -------
    ``auc``                pooled ROC AUC over all cells at risk.
    ``auc_within``         mean of the ROC AUC computed inside each
                           region. Not inflated by differences between
                           regions (large, diversified regions enter
                           more), so it measures the ability to rank the
                           options of a given region.
    ``average_precision``  area under the precision-recall curve (pooled);
                           informative when entries are rare.
    ``precision@k``        share of a region's k best-ranked options that
                           entered, averaged over regions.
    ``recall@k``           share of a region's entries found among its k
                           best-ranked options, averaged over the regions
                           that had entries.
    ``brier``              mean squared error of the score as a
                           probability; NaN unless all scores are in
                           [0, 1].

    Parameters
    ----------
    scores : array-like (R x C) or dict of them
        One predictor, or ``{name: matrix}`` to compare several. Labelled
        DataFrames are matched to `target` by label. NaN cells are left
        out for that predictor.
    target : array-like (R x C)
        1 = entered, 0 = at risk and did not enter, NaN = not at risk
        (first output of `entry_targets`).
    k : sequence of int
        Sizes of the per-region shortlists.
    baseline : str, optional
        Name of the predictor the others are compared with; adds the
        difference of each metric to the baseline.
    n_boot : int
        Number of bootstrap replications for 95 % intervals. Regions are
        resampled with replacement (cells of a region are not
        independent). 0 (default) skips the intervals.
    seed : int, optional
        Random seed of the bootstrap.

    Returns
    -------
    pd.DataFrame indexed by (score, metric) with column ``value``; plus
    ``lo`` and ``hi`` when ``n_boot > 0``; plus ``diff`` (and ``diff_lo``,
    ``diff_hi``) when a baseline is given. ``out["value"].unstack()``
    gives the score x metric table.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> target = pd.DataFrame([[0.0, 0.0, 1.0, 1.0]], index=["R1"], columns=list("ABCD"))
    >>> score = pd.DataFrame([[0.1, 0.4, 0.35, 0.8]], index=["R1"], columns=list("ABCD"))
    >>> ec.validate_entry(score, target, k=(2,))["value"].round(3).to_dict()[("score", "auc")]
    0.75
    """
    if not isinstance(scores, dict):
        scores = {"score": scores}
    if baseline is not None and baseline not in scores:
        raise ValueError(f"baseline {baseline!r} is not one of the scores {list(scores)}.")
    is_df = isinstance(target, pd.DataFrame)
    Y = np.array(target, dtype=float)
    ks = [int(v) for v in np.atleast_1d(k)]

    mats = {}
    for name, sc in scores.items():
        if is_df and isinstance(sc, pd.DataFrame):
            sc = sc.reindex(index=target.index, columns=target.columns)
        S = np.array(sc, dtype=float)
        if S.shape != Y.shape:
            raise ValueError(f"score {name!r} must be {Y.shape}; got {S.shape}.")
        mats[name] = S

    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, Y.shape[0], Y.shape[0]) for _ in range(n_boot)]

    point, boot = {}, {}
    for name, S in mats.items():
        Ys = np.where(np.isnan(S), np.nan, Y)
        per_region = _region_metrics(Ys, S, ks) + (np.nansum(Ys, axis=1),)
        point[name] = _metrics(Ys, S, ks, per_region)
        boot[name] = pd.DataFrame([_metrics(Ys, S, ks, per_region, rows) for rows in samples])

    metrics = list(next(iter(point.values())))
    out = pd.DataFrame(
        {"value": [point[name][metric] for name in mats for metric in metrics]},
        index=pd.MultiIndex.from_product([list(mats), metrics], names=["score", "metric"]),
    )

    def interval(draws, p):
        return np.concatenate([d.quantile(p).reindex(metrics).to_numpy() for d in draws])

    if n_boot:
        out["lo"] = interval(boot.values(), 0.025)
        out["hi"] = interval(boot.values(), 0.975)
    if baseline is not None:
        out["diff"] = out["value"] - np.tile(out["value"][baseline].to_numpy(), len(mats))
        if n_boot:
            diffs = [boot[name] - boot[baseline] for name in mats]
            out["diff_lo"] = interval(diffs, 0.025)
            out["diff_hi"] = interval(diffs, 0.975)
    return out
