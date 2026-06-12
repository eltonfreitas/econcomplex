"""
Stepping-stone forward model for ECI Optimization.

Calibrates the forecast model of specialization (eq. 1 of the paper),
computes the closed-form effort matrix W_cp (eq. 2), and projects the
future specialization matrix (no-policy baseline, W = 0).

References
----------
Stojkoski & Hidalgo (2026) "Optimizing economic complexity",
Research Policy 55, 105454.
Pinheiro et al. (2021) for relative relatedness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from ..core.utils import pivot_to_matrix, safe_divide
from ..core.rca import rca as compute_rca
from ..relatedness.density import relatedness_density

COEF_NAMES = ["b1_r_stepping", "b2_r_initial", "b3_relatedness",
              "b4_relative_relatedness", "b0_intercept"]


def _features(mat: pd.DataFrame, threshold: float, proximity_method: str):
    """RCA, log(1+RCA), relatedness (0-1) and relative relatedness for `mat`."""
    R = compute_rca(mat.values)
    r = np.log1p(R)
    dens = relatedness_density(mat, threshold=threshold,
                                proximity_method=proximity_method)
    omega = dens.values / 100.0  # 0-1 scale, as in the paper
    # Relative relatedness (Pinheiro et al. 2021, eq. 7): z-transform
    # against the statistics of the location's option set (RCA < threshold).
    # The same standardization is applied to specialized cells so the exit
    # model shares the scale of the entry model.
    opt = R < threshold
    n_opt = opt.sum(axis=1, keepdims=True)
    mu = safe_divide(np.where(opt, omega, 0.0).sum(axis=1, keepdims=True), n_opt)
    var = safe_divide(
        np.where(opt, (omega - mu) ** 2, 0.0).sum(axis=1, keepdims=True), n_opt
    )
    rel = safe_divide(omega - mu, np.sqrt(var))
    return R, r, omega, rel


def calibrate_steppingstone(
    df: pd.DataFrame,
    loc: str,
    act: str,
    val: str,
    time: str,
    horizon: int = 10,
    steppingstone: int = 5,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Dict:
    """
    Calibrate the stepping-stone forecast model (eq. 1 of Stojkoski &
    Hidalgo 2026) on a long-format panel.

    For every initial period t with t+steppingstone and t+horizon also
    present in the panel, fits by OLS:

      r_cp(t+horizon) = b1*r_cp(t+steppingstone) + b2*r_cp(t)
                        + b3*relatedness_cp(t) + b4*relative_relatedness_cp(t)
                        + b0

    where r = log(1 + RCA). Entry models use only cells with RCA(t) < 1;
    exit models use only cells with RCA(t) >= 1. Coefficients are averaged
    across all available initial periods.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel.
    loc, act, val, time : str
        Column names for location, activity, value, and (numeric) period.
    horizon : int
        Forecast horizon Delta-t in period units (default 10).
    steppingstone : int
        Steppingstone tau < horizon (default 5).
    threshold : float
        RCA binarization threshold for the relatedness features.
    proximity_method : str
        Proximity normalization ('max', 'sqrt', 'min').

    Returns
    -------
    dict with keys:
      'entry', 'exit' : coefficient dicts (b1, b2, b3, b4, b0)
      'horizon', 'steppingstone', 'initial_periods',
      'n_obs_entry', 'n_obs_exit'
    """
    if steppingstone >= horizon:
        raise ValueError("steppingstone must be smaller than horizon.")

    periods = sorted(df[time].unique())
    initial = [t for t in periods
               if (t + steppingstone) in periods and (t + horizon) in periods]
    if not initial:
        raise ValueError(
            f"No initial period t has both t+{steppingstone} and "
            f"t+{horizon} in the panel (periods available: {periods})."
        )

    coefs_entry: List[np.ndarray] = []
    coefs_exit: List[np.ndarray] = []
    n_entry = n_exit = 0

    for t in initial:
        mats = {}
        for y in (t, t + steppingstone, t + horizon):
            mats[y] = pivot_to_matrix(df[df[time] == y], loc, act, val)
        rows = mats[t].index
        cols = mats[t].columns
        for m in mats.values():
            rows = rows.union(m.index)
            cols = cols.union(m.columns)
        for y in mats:
            mats[y] = mats[y].reindex(index=rows, columns=cols, fill_value=0.0)

        R_t, r_t, omega, rel = _features(mats[t], threshold, proximity_method)
        r_tau = np.log1p(compute_rca(mats[t + steppingstone].values))
        r_T = np.log1p(compute_rca(mats[t + horizon].values))

        X = np.column_stack([
            r_tau.ravel(), r_t.ravel(), omega.ravel(), rel.ravel(),
            np.ones(r_t.size),
        ])
        y_vec = r_T.ravel()
        entry_mask = (R_t < threshold).ravel()

        for mask, store in ((entry_mask, coefs_entry), (~entry_mask, coefs_exit)):
            if mask.sum() > X.shape[1]:
                beta, *_ = np.linalg.lstsq(X[mask], y_vec[mask], rcond=None)
                store.append(beta)
        n_entry += int(entry_mask.sum())
        n_exit += int((~entry_mask).sum())

    if not coefs_entry or not coefs_exit:
        raise ValueError("Not enough observations to fit entry/exit models.")

    entry = dict(zip(COEF_NAMES, np.mean(coefs_entry, axis=0)))
    exit_ = dict(zip(COEF_NAMES, np.mean(coefs_exit, axis=0)))
    return {
        "entry": entry,
        "exit": exit_,
        "horizon": horizon,
        "steppingstone": steppingstone,
        "threshold": threshold,
        "proximity_method": proximity_method,
        "initial_periods": initial,
        "n_obs_entry": n_entry,
        "n_obs_exit": n_exit,
    }


def effort_matrix(
    mat: pd.DataFrame,
    model: Dict,
) -> pd.DataFrame:
    """
    Effort W_cp: the added RCA an economy must reach by the steppingstone
    period for the calibrated model to predict entry (RCA = 1) at the
    horizon (eq. 2 of Stojkoski & Hidalgo 2026, solved in closed form).

    Setting r_cp(t+horizon) = log(2) in the stepping-stone equation:

      W_cp = exp[(log 2 - b0 - b2*r - b3*omega - b4*rel) / b1] - 1 - RCA_cp

    Values are returned only for candidate cells (RCA < threshold);
    currently specialized cells are NaN. W <= 0 means the model already
    predicts entry without any boost.

    Parameters
    ----------
    mat : pd.DataFrame (R x C)
        Value matrix at the initial period.
    model : dict
        Output of `calibrate_steppingstone`.

    Returns
    -------
    R x C DataFrame of efforts.
    """
    b = model["entry"]
    if abs(b["b1_r_stepping"]) < 1e-12:
        raise ValueError(
            "The steppingstone coefficient (b1) of the entry model is ~0; "
            "the effort W_cp is undefined. Recalibrate the model (more "
            "periods or a different steppingstone/horizon)."
        )
    threshold = model.get("threshold", 1.0)
    R, r, omega, rel = _features(mat, threshold,
                                 model.get("proximity_method", "max"))

    numerator = (np.log(2.0) - b["b0_intercept"]
                 - b["b2_r_initial"] * r
                 - b["b3_relatedness"] * omega
                 - b["b4_relative_relatedness"] * rel)
    with np.errstate(over="ignore"):
        W = np.exp(numerator / b["b1_r_stepping"]) - 1.0 - R

    W = np.where(R < threshold, W, np.nan)
    return pd.DataFrame(W, index=mat.index, columns=mat.columns)


def forecast_specialization(
    mat: pd.DataFrame,
    model: Dict,
) -> Dict:
    """
    No-policy baseline forecast (W = 0): project RCA at t+horizon with the
    calibrated stepping-stone model, using entry coefficients for cells
    with RCA < threshold and exit coefficients otherwise.

    Parameters
    ----------
    mat : pd.DataFrame (R x C)
        Value matrix at the initial period.
    model : dict
        Output of `calibrate_steppingstone`.

    Returns
    -------
    dict with:
      'rca' : projected RCA matrix (DataFrame)
      'mcp' : projected binary specialization matrix (DataFrame)
      'pci' : projected PCI (Series, z-scored; NaN for activities trimmed
              from the projected matrix)
      'eci' : projection ECI per location = mean projected PCI over the
              projected portfolio (Series)
    """
    from ..complexity.eci_pci import eci_pci

    threshold = model.get("threshold", 1.0)
    R, r, omega, rel = _features(mat, threshold,
                                 model.get("proximity_method", "max"))

    r_hat = np.empty_like(r)
    for key, mask in (("entry", R < threshold), ("exit", R >= threshold)):
        b = model[key]
        pred = (b["b0_intercept"]
                + (b["b1_r_stepping"] + b["b2_r_initial"]) * r
                + b["b3_relatedness"] * omega
                + b["b4_relative_relatedness"] * rel)
        r_hat[mask] = pred[mask]

    R_hat = np.expm1(np.clip(r_hat, 0.0, None))
    rca_hat = pd.DataFrame(R_hat, index=mat.index, columns=mat.columns)
    mcp_hat = (rca_hat >= threshold).astype(float)

    _, pci = eci_pci(rca_hat, use_rca=False, threshold=threshold)
    pci = pci.rename("pci_projected")

    weights = mcp_hat.mul(pci.notna().astype(float), axis=1)
    portfolio = weights.values * np.nan_to_num(pci.values)[None, :]
    counts = weights.sum(axis=1).values
    eci_proj = pd.Series(
        safe_divide(portfolio.sum(axis=1), counts),
        index=mat.index, name="eci_projection",
    )
    eci_proj[counts == 0] = np.nan

    return {"rca": rca_hat, "mcp": mcp_hat, "pci": pci, "eci": eci_proj}
