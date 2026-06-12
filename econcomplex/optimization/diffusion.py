"""
Strategic diffusion on networks of related activities.

Implements the complex-contagion model of Alshamsi, Pinheiro & Hidalgo
(2018): the probability that a location activates a new activity grows as
a power of the fraction of related activities already present,

    p_i = B * (sum_j a_ij M_j / k_i) ** alpha          (eq. 1)

A diversification strategy is an ordered sequence of activation targets;
the expected waiting time of a target is 1/p_i given the current active
set. The paper shows that the greedy strategy (always target the most
related activity) is suboptimal: minimal total diversification time
requires targeting highly connected hubs during a narrow, relatively
early window ("suboptimality of relatedness").

References
----------
Alshamsi, Pinheiro & Hidalgo (2018) "Optimal diversification strategies
in the networks of related products and of related research areas",
Nature Communications 9, 1328.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

from ..core.utils import safe_divide, pivot_to_matrix
from ..core.rca import mcp as compute_mcp
from ..relatedness.proximity import proximity as compute_proximity

STRATEGIES = ("greedy", "majority", "high_degree", "low_degree", "random")


def proximity_network(
    mat: pd.DataFrame,
    phi: Optional[pd.DataFrame] = None,
    phi_threshold: float = 0.55,
    method: str = "max",
) -> pd.DataFrame:
    """
    Binary network of related activities: a_pp' = 1 if phi_pp' >= threshold.

    The default threshold (0.55) follows the product-space literature
    (Hidalgo et al. 2007). Pass a pre-computed `phi` to use another
    proximity measure.

    Returns
    -------
    C x C binary adjacency DataFrame with zero diagonal.
    """
    if phi is None:
        phi = compute_proximity(mat, method=method, compute="product")["product"]
    adj = (phi.values >= phi_threshold).astype(float)
    np.fill_diagonal(adj, 0.0)
    return pd.DataFrame(adj, index=phi.index, columns=phi.columns)


def activation_probabilities(
    adjacency: pd.DataFrame,
    active: Union[pd.Series, np.ndarray],
    B: float = 1.0,
    alpha: float = 1.0,
) -> pd.Series:
    """
    Activation probability of each inactive activity (eq. 1):
    p_i = B * (active neighbors / degree)^alpha; 0 for active nodes.
    """
    a = np.asarray(active, dtype=float)
    adj = adjacency.values
    k = adj.sum(axis=1)
    frac = safe_divide(adj @ a, k)
    p = B * frac ** alpha
    p[a.astype(bool)] = 0.0
    return pd.Series(p, index=adjacency.index, name="activation_probability")


def calibrate_contagion(
    df: pd.DataFrame,
    loc: str,
    act: str,
    val: str,
    time: str,
    adjacency: Optional[pd.DataFrame] = None,
    phi_threshold: float = 0.55,
    threshold: float = 1.0,
    presence_test: str = "rca",
    n_bins: int = 20,
) -> Dict:
    """
    Empirically calibrate B and alpha of the contagion model (eq. 1) from
    a panel: the probability that a location enters an activity between
    consecutive periods is regressed (log-log, on binned data) against
    the fraction of related activities already present.

    Alshamsi et al. (2018) report p ~ 0.16 x^1.03 for the product space
    and p ~ 0.74 x^1.09 for the research space.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel.
    loc, act, val, time : str
        Column names.
    adjacency : pd.DataFrame, optional
        Binary activity network. Computed from the first period via
        `proximity_network` if omitted.
    phi_threshold : float
        Proximity threshold when computing the network internally.
    threshold : float
        RCA binarization threshold for presences.
    presence_test : str
        Passed to `mcp` ('rca' default; use 'manual' when `val` is
        already a binary presence indicator).
    n_bins : int
        Number of bins of the related fraction.

    Returns
    -------
    dict with 'B', 'alpha', 'n_events', 'n_observations', and 'bins'
    (DataFrame with the binned fractions and empirical probabilities,
    useful for plotting the fit).
    """
    periods = sorted(df[time].unique())
    if len(periods) < 2:
        raise ValueError("Need at least 2 periods to observe entries.")

    mats = {t: pivot_to_matrix(df[df[time] == t], loc, act, val)
            for t in periods}
    if adjacency is None:
        adjacency = proximity_network(mats[periods[0]],
                                      phi_threshold=phi_threshold)
    acts = adjacency.index
    adj = adjacency.values
    k = adj.sum(axis=1)

    xs, ys = [], []
    for t0, t1 in zip(periods[:-1], periods[1:]):
        m0 = compute_mcp(mats[t0].reindex(columns=acts, fill_value=0.0),
                         presence_test=presence_test, rca_threshold=threshold)
        rows = m0.index
        m1 = compute_mcp(mats[t1].reindex(columns=acts, fill_value=0.0),
                         presence_test=presence_test, rca_threshold=threshold)
        m1 = m1.reindex(index=rows, fill_value=0.0)
        frac = safe_divide(m0.values @ adj, k[None, :])
        inactive = m0.values == 0
        xs.append(frac[inactive])
        ys.append(m1.values[inactive])

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, probs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x > lo) & (x <= hi)
        if sel.sum() > 0:
            centers.append(x[sel].mean())
            probs.append(y[sel].mean())
    bins = pd.DataFrame({"related_fraction": centers, "entry_probability": probs})

    fit = bins[(bins["related_fraction"] > 0) & (bins["entry_probability"] > 0)]
    if len(fit) < 2:
        raise ValueError("Not enough non-empty bins to fit the power law.")
    beta, intercept = np.polyfit(np.log(fit["related_fraction"]),
                                 np.log(fit["entry_probability"]), 1)

    return {
        "B": float(np.exp(intercept)),
        "alpha": float(beta),
        "n_events": int(y.sum()),
        "n_observations": int(len(y)),
        "bins": bins,
    }


def _pick_target(strategy, p, k, n_active_neighbors, rng):
    """Index of the next target among candidates (p > 0)."""
    cand = np.flatnonzero(p > 0)
    if cand.size == 0:
        return None
    if strategy == "greedy":
        return cand[np.argmax(p[cand])]
    if strategy == "majority":
        return cand[np.argmax(n_active_neighbors[cand])]
    if strategy == "high_degree":
        return cand[np.argmax(k[cand])]
    if strategy == "low_degree":
        return cand[np.argmin(k[cand])]
    if strategy == "random":
        return rng.choice(cand)
    raise ValueError(f"strategy must be one of {STRATEGIES}.")


def diversification_strategy(
    adjacency: pd.DataFrame,
    active0: Union[pd.Series, np.ndarray],
    B: float = 1.0,
    alpha: float = 1.0,
    strategy: str = "greedy",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a diversification sequence with one of the heuristic
    strategies of Alshamsi et al. (2018) and evaluate its expected times.

    At each step the strategy picks one potentially active target
    (an inactive activity with at least one active neighbor); the expected
    waiting time is 1/p_i (eq. 1) given the current active set, after
    which the target becomes active. Activities unreachable from the
    initial active set are not activated.

    Strategies: 'greedy' (highest activation probability — the most
    related), 'majority' (most active neighbors in absolute number),
    'high_degree', 'low_degree', and 'random'.

    Returns
    -------
    pd.DataFrame, one row per activated target in order, with columns
    [activity, expected_time, cumulative_time, degree, related_fraction].
    """
    rng = np.random.default_rng(seed)
    adj = adjacency.values
    k = adj.sum(axis=1)
    active = np.asarray(active0, dtype=float).copy()

    rows = []
    total = 0.0
    while True:
        n_act = adj @ active
        frac = safe_divide(n_act, k)
        p = B * frac ** alpha
        p[active.astype(bool)] = 0.0
        i = _pick_target(strategy, p, k, n_act, rng)
        if i is None:
            break
        total += 1.0 / p[i]
        active[i] = 1.0
        rows.append({
            "activity": adjacency.index[i],
            "expected_time": 1.0 / p[i],
            "cumulative_time": total,
            "degree": k[i],
            "related_fraction": frac[i],
        })

    return pd.DataFrame(
        rows, columns=["activity", "expected_time", "cumulative_time",
                       "degree", "related_fraction"],
    )


def expected_diversification_time(
    adjacency: pd.DataFrame,
    active0: Union[pd.Series, np.ndarray],
    sequence,
    B: float = 1.0,
    alpha: float = 1.0,
) -> float:
    """
    Expected total time of a fixed sequence of targets (activity labels).
    Returns inf if some target has no active neighbor when reached.
    """
    adj = adjacency.values
    k = adj.sum(axis=1)
    idx = adjacency.index.get_indexer(pd.Index(sequence))
    if (idx < 0).any():
        raise ValueError("sequence contains activities absent from the network.")
    active = np.asarray(active0, dtype=float).copy()
    total = 0.0
    for i in idx:
        cnt = adj[i] @ active
        if cnt == 0 or k[i] == 0:
            return np.inf
        total += 1.0 / (B * (cnt / k[i]) ** alpha)
        active[i] = 1.0
    return total


def compare_strategies(
    adjacency: pd.DataFrame,
    active0: Union[pd.Series, np.ndarray],
    B: float = 1.0,
    alpha: float = 1.0,
    n_random: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Expected total diversification time of each heuristic strategy
    ('random' averaged over `n_random` runs).

    Returns
    -------
    pd.DataFrame indexed by strategy with columns
    [total_time, n_activated].
    """
    rows = {}
    for s in STRATEGIES:
        if s == "random":
            runs = [diversification_strategy(adjacency, active0, B, alpha,
                                             strategy=s, seed=seed + r)
                    for r in range(n_random)]
            rows[s] = {
                "total_time": float(np.mean(
                    [r_["cumulative_time"].iloc[-1] for r_ in runs if len(r_)])),
                "n_activated": int(np.mean([len(r_) for r_ in runs])),
            }
        else:
            r_ = diversification_strategy(adjacency, active0, B, alpha, strategy=s)
            rows[s] = {
                "total_time": float(r_["cumulative_time"].iloc[-1]) if len(r_) else np.nan,
                "n_activated": len(r_),
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def optimize_sequence(
    adjacency: pd.DataFrame,
    active0: Union[pd.Series, np.ndarray],
    B: float = 1.0,
    alpha: float = 1.0,
    n_iter: int = 2000,
    seed: int = 0,
) -> Dict:
    """
    Approximate the optimal diversification sequence by simulated
    annealing over target orderings (pairwise-swap proposals), starting
    from the greedy sequence.

    Exact optimality is combinatorial (Alshamsi et al. solve it in closed
    form only for stylized networks); annealing reproduces the paper's
    qualitative optimum — targeting hubs earlier than the greedy strategy
    does — and never returns a sequence worse than greedy.

    Returns
    -------
    dict with 'sequence' (list of activity labels), 'total_time',
    'greedy_time', and 'improvement' (greedy_time - total_time).
    """
    rng = np.random.default_rng(seed)
    greedy = diversification_strategy(adjacency, active0, B, alpha, "greedy")
    order = list(greedy["activity"])
    if len(order) < 2:
        t = float(greedy["cumulative_time"].iloc[-1]) if len(order) else 0.0
        return {"sequence": order, "total_time": t,
                "greedy_time": t, "improvement": 0.0}

    def cost(seq):
        return expected_diversification_time(adjacency, active0, seq, B, alpha)

    current = order[:]
    current_cost = cost(current)
    greedy_cost = current_cost
    best, best_cost = current[:], current_cost

    t0, tf = 0.05 * current_cost, 1e-4 * current_cost
    cooling = (tf / t0) ** (1.0 / max(n_iter, 1))
    temp = t0
    n = len(order)

    for _ in range(n_iter):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        proposal = current[:]
        proposal[i], proposal[j] = proposal[j], proposal[i]
        c = cost(proposal)
        if c < current_cost or rng.random() < np.exp(-(c - current_cost) / temp):
            current, current_cost = proposal, c
            if c < best_cost:
                best, best_cost = proposal, c
        temp *= cooling

    return {
        "sequence": best,
        "total_time": float(best_cost),
        "greedy_time": float(greedy_cost),
        "improvement": float(greedy_cost - best_cost),
    }
