"""
ECI Optimization: 0-1 portfolio selection.

Selects the minimal-effort set of new specializations that raises a
location's projection ECI (mean PCI of its portfolio) to a target.

References
----------
Stojkoski & Hidalgo (2026) "Optimizing economic complexity",
Research Policy 55, 105454.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

from .steppingstone import effort_matrix, forecast_specialization

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except ImportError:  # SciPy < 1.9
    _HAS_MILP = False


def _select_portfolio(costs: np.ndarray, surplus: np.ndarray, deficit: float,
                      solver: str) -> Optional[np.ndarray]:
    """
    Minimize sum(costs[x]) s.t. sum(surplus[x]) >= deficit, x binary.
    Returns a boolean mask over the candidates, or None if infeasible.
    """
    if surplus.sum() < deficit:
        return None

    if solver == "milp" and _HAS_MILP:
        res = milp(
            c=costs,
            constraints=LinearConstraint(surplus[None, :], lb=deficit),
            integrality=np.ones_like(costs),
            bounds=Bounds(0, 1),
        )
        if res.status == 0 and res.x is not None:
            return res.x > 0.5
        return None

    # Greedy fallback: cheapest cost per unit of ECI surplus first
    order = np.argsort(costs / surplus)
    chosen = np.zeros(len(costs), dtype=bool)
    covered = 0.0
    for i in order:
        if covered >= deficit:
            break
        chosen[i] = True
        covered += surplus[i]
    return chosen if covered >= deficit else None


def eci_optimization(
    mat: pd.DataFrame,
    model: Dict,
    delta_eci: float = 0.1,
    target_eci: Optional[Union[float, Dict, pd.Series]] = None,
    locations: Optional[list] = None,
    solver: str = "milp",
) -> pd.DataFrame:
    """
    ECI Optimization (Stojkoski & Hidalgo 2026): identify, per location,
    the minimal-effort portfolio of new specializations that raises the
    projected ECI to a target.

    The pipeline: (i) project the no-policy specialization matrix and PCI
    at t+horizon (`forecast_specialization`); (ii) compute the effort
    W_cp required for each candidate entry (`effort_matrix`); (iii) solve
    the 0-1 program

        min sum_p W_cp x_cp
        s.t. mean PCI of (projected portfolio + selected) >= target ECI

    linearized as sum_p (PCI_p - target) x_cp >= deficit and solved
    exactly with `scipy.optimize.milp` (greedy fallback on SciPy < 1.9).

    Parameters
    ----------
    mat : pd.DataFrame (R x C)
        Value matrix at the initial period.
    model : dict
        Output of `calibrate_steppingstone`.
    delta_eci : float
        Target increase over each location's projected ECI (default 0.1,
        in PCI standard-deviation units). Ignored when `target_eci` given.
    target_eci : float, dict, or pd.Series, optional
        Absolute ECI target (single value or per location).
    locations : list, optional
        Subset of locations to optimize (default: all rows of `mat`).
    solver : str
        'milp' (exact, default) or 'greedy'.

    Returns
    -------
    pd.DataFrame with one row per suggested activity:
      [location, activity, effort, pci_projected, eci_projected,
       eci_target, eci_achieved]
    Locations whose projected ECI already meets the target contribute no
    rows; infeasible locations are skipped with a warning.
    """
    forecast = forecast_specialization(mat, model)
    W = effort_matrix(mat, model)
    pci = forecast["pci"]
    mcp_hat = forecast["mcp"]
    eci_proj = forecast["eci"]

    locs = locations if locations is not None else list(mat.index)
    valid_p = pci.notna()
    rows = []

    for c in locs:
        base_eci = eci_proj.get(c, np.nan)
        if np.isnan(base_eci):
            warnings.warn(f"Location {c!r}: empty projected portfolio; skipped.")
            continue

        if target_eci is None:
            target = base_eci + delta_eci
        elif np.isscalar(target_eci):
            target = float(target_eci)
        else:
            target = float(pd.Series(target_eci).get(c, np.nan))
            if np.isnan(target):
                warnings.warn(f"Location {c!r}: no target provided; skipped.")
                continue

        m_row = mcp_hat.loc[c].values.astype(bool) & valid_p.values
        deficit = -float(((pci[valid_p] - target) * mcp_hat.loc[c][valid_p]).sum())
        if deficit <= 0:
            continue  # target already met by the projected portfolio

        # Candidates: not in projected portfolio, PCI above target
        # (others cannot help the constraint), finite positive-side effort
        cand = (~m_row) & valid_p.values & (pci.values > target) \
            & np.isfinite(np.nan_to_num(W.loc[c].values, nan=np.inf))
        if not cand.any():
            warnings.warn(f"Location {c!r}: no feasible candidates; skipped.")
            continue

        costs = np.clip(W.loc[c].values[cand], 0.0, None)
        surplus = (pci.values - target)[cand]
        chosen = _select_portfolio(costs, surplus, deficit, solver)
        if chosen is None:
            warnings.warn(
                f"Location {c!r}: target ECI {target:.3f} infeasible with "
                "the available candidates; skipped."
            )
            continue

        acts = mat.columns[cand][chosen]
        n_new = chosen.sum()
        achieved = (
            (pci[valid_p] * mcp_hat.loc[c][valid_p]).sum()
            + pci[acts].sum()
        ) / (mcp_hat.loc[c][valid_p].sum() + n_new)

        for a in acts:
            rows.append({
                "location": c,
                "activity": a,
                "effort": float(W.loc[c, a]),
                "pci_projected": float(pci[a]),
                "eci_projected": float(base_eci),
                "eci_target": target,
                "eci_achieved": float(achieved),
            })

    return pd.DataFrame(
        rows, columns=["location", "activity", "effort", "pci_projected",
                       "eci_projected", "eci_target", "eci_achieved"],
    )
