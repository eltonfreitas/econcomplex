"""Tests for the ECI Optimization module (Stojkoski & Hidalgo 2026)."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import econcomplex as ec


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def panel():
    """Three-period panel (2000, 2005, 2010) with persistent structure."""
    rng = np.random.default_rng(0)
    df0 = ec.make_sample_data(n_locs=30, n_acts=20, seed=0)
    mats = {2000: ec.pivot_to_matrix(df0, "loc", "act", "val")}
    for y0, y1 in [(2000, 2005), (2005, 2010)]:
        prev = mats[y0]
        noise = rng.lognormal(mean=0.0, sigma=0.3, size=prev.shape)
        mats[y1] = prev * noise * 1.1
    frames = []
    for y, m in mats.items():
        d = ec.melt_matrix(m, "loc", "act", "val")
        d = d[d["val"] > 0].copy()
        d["year"] = y
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def model(panel):
    return ec.calibrate_steppingstone(
        panel, "loc", "act", "val", "year", horizon=10, steppingstone=5
    )


@pytest.fixture(scope="module")
def mat_latest(panel):
    return ec.pivot_to_matrix(panel[panel["year"] == 2010], "loc", "act", "val")


# ── Stepping-stone calibration ────────────────────────────────────────────────
def test_calibrate_returns_entry_and_exit(model):
    for key in ("entry", "exit"):
        coefs = model[key]
        assert set(coefs) == {"b1_r_stepping", "b2_r_initial",
                              "b3_relatedness", "b4_relative_relatedness",
                              "b0_intercept"}
        assert all(np.isfinite(v) for v in coefs.values())
    assert model["n_obs_entry"] > 0 and model["n_obs_exit"] > 0
    assert model["initial_periods"] == [2000]


def test_calibrate_persistence_positive(model):
    """RCA is persistent in the synthetic panel, so the steppingstone
    coefficient of the exit model must be positive."""
    assert model["exit"]["b1_r_stepping"] > 0


def test_calibrate_requires_compatible_periods(panel):
    with pytest.raises(ValueError):
        ec.calibrate_steppingstone(panel, "loc", "act", "val", "year",
                                   horizon=7, steppingstone=3)


# ── Effort matrix ─────────────────────────────────────────────────────────────
def test_effort_nan_for_specialized(mat_latest, model):
    W = ec.effort_matrix(mat_latest, model)
    R = ec.rca(mat_latest)
    assert W.shape == mat_latest.shape
    assert W.values[R.values >= 1].size == np.isnan(W.values[R.values >= 1]).sum()
    cand = W.values[R.values < 1]
    assert np.isfinite(cand).any()


def test_effort_decreases_with_rca(mat_latest, model):
    """Cells closer to specialization require less effort, all else equal:
    W + RCA is what the model demands at the steppingstone, so W must be
    weakly decreasing in current RCA for fixed relatedness terms."""
    W = ec.effort_matrix(mat_latest, model)
    R = ec.rca(mat_latest)
    mask = (R.values < 1) & np.isfinite(W.values)
    corr = np.corrcoef(R.values[mask], W.values[mask])[0, 1]
    assert corr < 0


# ── Forecast ──────────────────────────────────────────────────────────────────
def test_forecast_outputs(mat_latest, model):
    fc = ec.forecast_specialization(mat_latest, model)
    assert fc["rca"].shape == mat_latest.shape
    assert set(np.unique(fc["mcp"].values)).issubset({0.0, 1.0})
    assert (fc["rca"].values >= 0).all()
    assert fc["pci"].notna().sum() > 0
    assert fc["eci"].notna().sum() > 0


def test_forecast_matches_model_equation(mat_latest, model):
    """The no-policy forecast must equal the calibrated equation with the
    steppingstone term collapsed to r(t) (eq. 2 with W = 0)."""
    from econcomplex.optimization.steppingstone import _features

    R, r, omega, rel = _features(mat_latest, model["threshold"],
                                 model["proximity_method"])
    fc = ec.forecast_specialization(mat_latest, model)
    for key, mask in (("entry", R < 1.0), ("exit", R >= 1.0)):
        b = model[key]
        expected = np.expm1(np.clip(
            b["b0_intercept"]
            + (b["b1_r_stepping"] + b["b2_r_initial"]) * r
            + b["b3_relatedness"] * omega
            + b["b4_relative_relatedness"] * rel, 0.0, None))
        assert np.allclose(fc["rca"].values[mask], expected[mask]), key


# ── 0-1 optimization ──────────────────────────────────────────────────────────
def test_eci_optimization_reaches_target(mat_latest, model):
    result = ec.eci_optimization(mat_latest, model, delta_eci=0.1)
    assert len(result) > 0
    assert (result["eci_achieved"] >= result["eci_target"] - 1e-9).all()
    # suggested activities must be outside the projected portfolio
    fc = ec.forecast_specialization(mat_latest, model)
    for _, row in result.iterrows():
        assert fc["mcp"].loc[row["location"], row["activity"]] == 0.0
    # and have PCI above the target (otherwise they cannot help)
    assert (result["pci_projected"] > result["eci_target"]).all()


def test_eci_optimization_minimizes_effort(mat_latest, model):
    """The exact 0-1 solution cannot cost more than the greedy one."""
    exact = ec.eci_optimization(mat_latest, model, delta_eci=0.1, solver="milp")
    greedy = ec.eci_optimization(mat_latest, model, delta_eci=0.1, solver="greedy")
    cost_exact = exact.groupby("location")["effort"].apply(
        lambda s: np.clip(s, 0, None).sum())
    cost_greedy = greedy.groupby("location")["effort"].apply(
        lambda s: np.clip(s, 0, None).sum())
    common = cost_exact.index.intersection(cost_greedy.index)
    assert len(common) > 0
    assert (cost_exact[common] <= cost_greedy[common] + 1e-9).all()


def test_eci_optimization_absolute_target(mat_latest, model):
    fc = ec.forecast_specialization(mat_latest, model)
    target = float(fc["eci"].max()) + 0.05
    result = ec.eci_optimization(mat_latest, model, target_eci=target)
    if len(result):
        assert (result["eci_target"] == target).all()


def test_eci_optimization_skips_infeasible(mat_latest, model):
    with pytest.warns(UserWarning):
        result = ec.eci_optimization(mat_latest, model, delta_eci=50.0)
    assert len(result) == 0


# ── Growth targeting ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def growth_model():
    rng = np.random.default_rng(1)
    locs = [f"L{i:02d}" for i in range(40)]
    rows = []
    gdppc = pd.Series(rng.uniform(2e3, 5e4, len(locs)), index=locs)
    eci = pd.Series(rng.normal(0, 1, len(locs)), index=locs)
    for year in (2000, 2010, 2020):
        for c in locs:
            rows.append({"loc": c, "year": year,
                         "gdppc": gdppc[c], "eci": eci[c]})
        z = (np.log(gdppc) - np.log(gdppc).mean()) / np.log(gdppc).std()
        growth = 0.02 + 0.01 * eci - 0.005 * z \
            + rng.normal(0, 0.002, len(locs))
        gdppc = gdppc * np.exp(10 * growth)
        eci = eci + rng.normal(0, 0.1, len(locs))
    df = pd.DataFrame(rows)
    return ec.calibrate_growth_model(df, "loc", "year", "gdppc", "eci",
                                     horizon=10), df


def test_growth_model_recovers_signs(growth_model):
    model, _ = growth_model
    assert model["a1_eci"] > 0          # complexity helps growth
    assert model["a2_z_gdppc"] < 0      # convergence
    assert model["n_obs"] > 0


def test_growth_target_roundtrip(growth_model):
    model, _ = growth_model
    gdp_now = 15000.0
    target_growth = 0.035
    eci_star = ec.eci_target_for_growth(model, target_growth, gdp_now)
    back = ec.expected_growth(model, eci_star, gdp_now)
    assert np.isclose(back, target_growth, atol=1e-10)


def test_growth_target_series(growth_model):
    model, _ = growth_model
    gdp_now = pd.Series([5000.0, 30000.0], index=["A", "B"])
    eci_star = ec.eci_target_for_growth(model, 0.03, gdp_now)
    assert isinstance(eci_star, pd.Series)
    assert list(eci_star.index) == ["A", "B"]
