"""
Growth targeting for ECI Optimization.

Calibrates the growth regression (eq. 3 of Stojkoski & Hidalgo 2026) and
inverts it to find the ECI compatible with a target growth rate, which can
then be fed to `eci_optimization` as `target_eci`.

References
----------
Stojkoski & Hidalgo (2026) "Optimizing economic complexity",
Research Policy 55, 105454.
Hausmann et al. (2014) "The Atlas of Economic Complexity".
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def calibrate_growth_model(
    df: pd.DataFrame,
    loc: str,
    time: str,
    gdppc: str,
    eci: str,
    horizon: int = 10,
) -> Dict:
    """
    Calibrate the panel growth regression (eq. 3 of Stojkoski & Hidalgo
    2026) by OLS:

      annualized log growth of GDPpc over `horizon`
        = a1*ECI + a2*z + a3*(ECI x z) + period fixed effects + u

    where z is the log of initial GDP per capita, z-score normalized
    across locations within each initial period (Solow convergence term).

    Parameters
    ----------
    df : pd.DataFrame
        Long format with one row per (location, period).
    loc, time, gdppc, eci : str
        Column names for location, (numeric) period, GDP per capita, and
        ECI.
    horizon : int
        Growth horizon in period units (default 10).

    Returns
    -------
    dict with keys 'a1_eci', 'a2_z_gdppc', 'a3_interaction',
    'period_effects' (dict period -> fixed effect), 'z_stats'
    (dict period -> (mean, std) of log GDPpc), 'horizon', 'n_obs'.
    """
    periods = sorted(df[time].unique())
    initial = [t for t in periods if (t + horizon) in periods]
    if not initial:
        raise ValueError(
            f"No initial period t has t+{horizon} in the panel "
            f"(periods available: {periods})."
        )

    obs_y, obs_eci, obs_z, obs_t = [], [], [], []
    z_stats = {}

    for t in initial:
        d0 = df[df[time] == t].set_index(loc)
        d1 = df[df[time] == t + horizon].set_index(loc)
        common = d0.index.intersection(d1.index)
        d0, d1 = d0.loc[common], d1.loc[common]
        ok = (d0[gdppc] > 0) & (d1[gdppc] > 0) & d0[eci].notna()
        if ok.sum() < 5:
            continue
        lg = np.log(d0.loc[ok, gdppc].astype(float))
        mu, sd = lg.mean(), lg.std()
        if sd == 0:
            continue
        z_stats[t] = (float(mu), float(sd))
        z = (lg - mu) / sd
        growth = (np.log(d1.loc[ok, gdppc].astype(float)).values - lg.values) / horizon
        obs_y.append(growth)
        obs_eci.append(d0.loc[ok, eci].astype(float).values)
        obs_z.append(z.values)
        obs_t.append(np.full(ok.sum(), t))

    if not obs_y:
        raise ValueError("Not enough observations to fit the growth model.")

    y = np.concatenate(obs_y)
    e = np.concatenate(obs_eci)
    z = np.concatenate(obs_z)
    t_arr = np.concatenate(obs_t)
    used_periods = sorted(set(t_arr))

    # Design: [ECI, z, ECI*z, one dummy per initial period (no intercept)]
    dummies = np.column_stack([(t_arr == t).astype(float) for t in used_periods])
    X = np.column_stack([e, z, e * z, dummies])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "a1_eci": float(beta[0]),
        "a2_z_gdppc": float(beta[1]),
        "a3_interaction": float(beta[2]),
        "period_effects": {t: float(b) for t, b in zip(used_periods, beta[3:])},
        "z_stats": z_stats,
        "horizon": horizon,
        "n_obs": int(len(y)),
    }


def _z_and_gamma(model: Dict, gdppc_now, period=None):
    """z-score of log GDPpc and period effect for prediction."""
    ref = period if period is not None else max(model["z_stats"])
    mu, sd = model["z_stats"][ref]
    z = (np.log(np.asarray(gdppc_now, dtype=float)) - mu) / sd
    gamma = model["period_effects"][ref]
    return z, gamma


def expected_growth(
    model: Dict,
    eci: Union[float, pd.Series],
    gdppc_now: Union[float, pd.Series],
    period=None,
) -> Union[float, pd.Series]:
    """
    Annualized log growth rate implied by the calibrated model for a given
    ECI and current GDP per capita (using the most recent period's fixed
    effect unless `period` is given).
    """
    z, gamma = _z_and_gamma(model, gdppc_now, period)
    e = eci.values if isinstance(eci, pd.Series) else np.asarray(eci, dtype=float)
    g = model["a1_eci"] * e + model["a2_z_gdppc"] * z \
        + model["a3_interaction"] * e * z + gamma
    if isinstance(eci, pd.Series):
        return pd.Series(g, index=eci.index, name="expected_growth")
    return float(g) if np.ndim(g) == 0 else g


def eci_target_for_growth(
    model: Dict,
    growth_target: float,
    gdppc_now: Union[float, pd.Series],
    period=None,
) -> Union[float, pd.Series]:
    """
    Invert the growth regression to find the ECI compatible with a target
    annualized log growth rate (e.g. 0.035 for ~3.5 % per year):

      ECI* = (growth - a2*z - gamma) / (a1 + a3*z)

    The result can be passed to `eci_optimization` as `target_eci`.
    """
    import warnings

    z, gamma = _z_and_gamma(model, gdppc_now, period)
    denom = np.asarray(model["a1_eci"] + model["a3_interaction"] * z, dtype=float)
    near_zero = np.abs(denom) < 1e-12
    if np.any(near_zero):
        warnings.warn(
            "eci_target_for_growth: the marginal effect of ECI "
            "(a1 + a3*z) is ~0 for some inputs; the target is undefined "
            "there and returned as NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        denom = np.where(near_zero, np.nan, denom)
    target = (growth_target - model["a2_z_gdppc"] * z - gamma) / denom
    if isinstance(gdppc_now, pd.Series):
        return pd.Series(target, index=gdppc_now.index, name="eci_target")
    return float(target) if np.ndim(target) == 0 else target
