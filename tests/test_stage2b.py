"""Tests of the second batch of 1.2.0: GENEPY, ECI diagnostics, implied
RCA, opportunity alignment, potential employment and the optional
machine-learning prediction of entry."""

import numpy as np
import pandas as pd
import pytest

import econcomplex as ec


@pytest.fixture
def V():
    """RCA = [[2, 0, 0], [0, 2, 2], [1, 1, 1]]  ->  M = [[1,0,0],[0,1,1],[1,1,1]]"""
    return pd.DataFrame(
        [[4.0, 0.0, 0.0], [0.0, 2.0, 2.0], [4.0, 2.0, 2.0]],
        index=["R1", "R2", "R3"], columns=["A", "B", "C"],
    )


@pytest.fixture
def big():
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=42)
    return ec.pivot_to_matrix(df, "loc", "act", "val")


# ── GENEPY ────────────────────────────────────────────────────────────────────
def _genepy_like_the_authors(M, group):
    """Line-by-line transcription of genepy_index.m (Zenodo 3876721)."""
    kc = M.sum(axis=1, keepdims=True)
    RW = M / kc
    kp_1 = RW.sum(axis=0, keepdims=True)
    W = M / (kc @ kp_1)
    P = W @ W.T if group == "countries" else W.T @ W
    P[np.eye(len(P)) != 0] = 0
    evalues, evectors = np.linalg.eig(P)
    order = np.argsort(-np.abs(evalues), kind="stable")
    evalues, evectors = evalues[order].real, evectors[:, order].real
    E = np.column_stack([np.abs(evectors[:, 0]), evectors[:, 1]])
    lam = evalues[:2]
    return ((E ** 2) @ lam) ** 2 + 2 * (E ** 2) @ (lam ** 2)


def test_genepy_matches_the_authors_code(big):
    M = ec.mcp(big).to_numpy(dtype=float)
    regions, activities = ec.genepy(big)
    np.testing.assert_allclose(regions, _genepy_like_the_authors(M, "countries"), rtol=1e-8)
    np.testing.assert_allclose(activities, _genepy_like_the_authors(M, "products"), rtol=1e-8)
    assert (regions >= 0).all() and list(regions.index) == list(big.index)
    # labels follow the units when the rows are reordered
    shuffled = big.sample(frac=1, random_state=3)
    np.testing.assert_allclose(ec.genepy(shuffled)[0][big.index], regions, rtol=1e-8)


def test_genepy_through_eci_pci_trims(V):
    g, a = ec.eci_pci(V, method="genepy")
    np.testing.assert_allclose(g, ec.genepy(V)[0])
    wider = V.copy()
    wider.loc["R4"] = 0.0
    g4, _ = ec.eci_pci(wider, method="genepy")
    assert np.isnan(g4["R4"])
    np.testing.assert_allclose(g4[["R1", "R2", "R3"]], g)
    with pytest.raises(ValueError, match="zero diversity"):
        ec.genepy(wider)


# ── ECI diagnostics ───────────────────────────────────────────────────────────
def test_eci_diagnostics_clear_vs_blurred_structure():
    rng = np.random.default_rng(0)
    # two well separated groups of regions: a clean second eigenvector
    blocks = np.zeros((30, 20))
    blocks[:15, :10] = 1
    blocks[15:, 10:] = 1
    blocks[:15, 10:] = rng.random((15, 10)) < 0.15
    blocks[15:, :10] = rng.random((15, 10)) < 0.60     # second group is also more diverse
    noise = (rng.random((30, 20)) < 0.4).astype(float)
    kw = {"use_rca": False, "threshold": 0.5, "n_boot": 40, "seed": 1}
    clear, blurred = ec.eci_diagnostics(blocks, **kw), ec.eci_diagnostics(noise, **kw)
    assert clear["gap"] > 3 * blurred["gap"]
    assert clear["rank_stability_mean"] > 0.9 > blurred["rank_stability_mean"]
    assert clear["sign_flips"] <= 0.05 and blurred["sign_flips"] > clear["sign_flips"]
    assert clear["eigenvalue_2"] < 1 and clear["eigenvalue_2"] > clear["eigenvalue_3"]
    assert clear["n_regions"] == 30 and clear["n_activities"] == 20
    pd.testing.assert_series_equal(clear, ec.eci_diagnostics(blocks, **kw))
    quick = ec.eci_diagnostics(blocks, use_rca=False, threshold=0.5, n_boot=0)
    assert np.isnan(quick["rank_stability_mean"]) and quick["gap"] == pytest.approx(clear["gap"])


def test_eci_diagnostics_correlation_with_diversity(big):
    d = ec.eci_diagnostics(big, n_boot=0)
    eci, _ = ec.eci_pci(big)
    assert d["corr_diversity"] == pytest.approx(np.corrcoef(eci, ec.diversity(big))[0, 1])
    assert d["corr_diversity"] > 0            # sign convention of the ECI


# ── implied RCA ───────────────────────────────────────────────────────────────
def test_implied_rca_nearest_neighbour_by_hand():
    # A and B move together across regions, C moves against them
    r = pd.DataFrame([[1.0, 2.0, 9.0], [2.0, 4.0, 5.0], [3.0, 6.0, 4.0], [4.0, 8.0, 1.0]],
                     index=list("wxyz"), columns=list("ABC"))
    imp = ec.implied_rca(r, space="activity", k=1, cap=None, use_rca=False)
    np.testing.assert_allclose(imp["A"], r["B"])      # nearest activity of A is B
    np.testing.assert_allclose(imp["B"], r["A"])
    # with two neighbours: weights (1 + corr) / 2, normalized
    imp2 = ec.implied_rca(r, space="activity", k=2, cap=None, use_rca=False)
    s_ab, s_ac = 1.0, (1 + np.corrcoef(r["A"], r["C"])[0, 1]) / 2
    np.testing.assert_allclose(imp2["A"], (s_ab * r["B"] + s_ac * r["C"]) / (s_ab + s_ac))
    # region space, one neighbour: x and y have the most similar profiles to each other?
    reg = ec.implied_rca(r, space="region", k=1, cap=None, use_rca=False)
    sim = np.corrcoef(r.values)
    np.fill_diagonal(sim, -np.inf)
    np.testing.assert_allclose(reg.values, r.values[sim.argmax(axis=1)])
    both = ec.implied_rca(r, space="mean", k=1, cap=None, use_rca=False)
    np.testing.assert_allclose(both.values, (imp.values + reg.values) / 2)


def test_implied_rca_duplicated_region(big):
    # a copy of a region (the library computes the RCA, so duplicate after it)
    r = ec.rca(big)
    twin = pd.concat([r, r.iloc[[4]].rename(index=lambda s: s + "_copy")])
    for space in ("activity", "region", "mean"):
        imp = ec.implied_rca(twin, space=space, use_rca=False)
        np.testing.assert_allclose(imp.iloc[-1], imp.iloc[4], err_msg=space)
    assert ec.implied_rca(big.values).shape == big.shape
    # the cap only changes the similarities: no cap and a huge cap coincide
    np.testing.assert_allclose(ec.implied_rca(big, cap=None), ec.implied_rca(big, cap=1e9))
    with pytest.raises(ValueError):
        ec.implied_rca(big, space="product")


# ── opportunity alignment ─────────────────────────────────────────────────────
def test_opportunity_alignment():
    pci = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0})
    feas = pd.DataFrame(
        [[40.0, 30.0, 20.0, 10.0],        # nearby = valuable
         [10.0, 20.0, 30.0, 40.0],        # nearby = low value
         [np.nan, 5.0, np.nan, 9.0],      # two options only
         [7.0, 7.0, 7.0, 7.0]],           # no variation
        index=["up", "trap", "few", "flat"], columns=list("ABCD"))
    a = ec.opportunity_alignment(feas, pci)
    assert a["up"] == pytest.approx(1) and a["trap"] == pytest.approx(-1)
    assert np.isnan(a["few"]) and np.isnan(a["flat"])
    assert ec.opportunity_alignment(feas, pci, min_options=2)["few"] == pytest.approx(-1)
    pear = ec.opportunity_alignment(feas.values, pci.values, method="pearson")
    assert pear[0] == pytest.approx(1)
    skew = feas.copy()
    skew.loc["up"] = [100.0, 3.0, 2.0, 1.0]           # same ranks, not linear
    assert ec.opportunity_alignment(skew, pci)["up"] == pytest.approx(1)
    assert ec.opportunity_alignment(skew, pci, method="pearson")["up"] < 0.9


# ── potential employment ──────────────────────────────────────────────────────
def test_potential_employment_lq1_and_entrants():
    values = pd.DataFrame([[30.0, 10.0, 0.0], [20.0, 40.0, 100.0]],
                          index=["R1", "R2"], columns=list("ABC"))
    prob = pd.DataFrame([[np.nan, 1.0, 0.25], [np.nan, np.nan, np.nan]],
                        index=values.index, columns=values.columns)
    out = ec.potential_employment(prob, values)
    assert list(out.index.names) == ["region", "activity", "scenario"] and len(out) == 2
    # with probability 1: E_r E_c / E
    assert out.loc[("R1", "B", "lq1"), "expected"] == pytest.approx(40 * 50 / 200)
    assert out.loc[("R1", "C", "lq1"), "expected"] == pytest.approx(0.25 * 40 * 100 / 200)

    entrants = pd.DataFrame([[np.nan, 10.0, np.nan], [np.nan, 30.0, np.nan]],
                            index=values.index, columns=values.columns)
    q = ec.potential_employment(prob, values, arrival="entrants", entrants=entrants)
    assert set(q.index.get_level_values("scenario")) == {"q25", "q50", "q75"}
    assert q.loc[("R1", "B", "q50"), "arrival"] == pytest.approx(20)
    assert q.loc[("R1", "B", "q25"), "expected"] == pytest.approx(15)
    assert np.isnan(q.loc[("R1", "C", "q50"), "expected"])      # no entrant observed in C
    with pytest.raises(ValueError):
        ec.potential_employment(prob, values, arrival="entrants")


# ── features ──────────────────────────────────────────────────────────────────
def test_build_features_known(V):
    dens = ec.relatedness_density(V)
    wage = pd.DataFrame({"wage": [3.0, 1.0, 2.0]}, index=["C", "A", "B"])
    pop = pd.DataFrame({"pop": [10.0, 20.0, 30.0]}, index=["R3", "R2", "R1"])
    X = ec.build_features(V, {"density": dens}, region_vars=pop, activity_vars=wage)
    assert list(X.index.names) == ["region", "activity"] and len(X) == 9
    assert list(X.columns) == ["density", "rca", "log_region_size", "diversity",
                               "log_activity_size", "ubiquity", "pop", "wage"]
    row = X.loc[("R2", "A")]
    assert row["density"] == dens.loc["R2", "A"] and row["rca"] == 0
    assert row["log_region_size"] == pytest.approx(np.log1p(4))
    assert row["diversity"] == 2 and row["ubiquity"] == 2
    assert row["log_activity_size"] == pytest.approx(np.log1p(8))
    assert row["pop"] == 20 and row["wage"] == 1
    at_risk = ec.rca(V) < 0.5
    Xr = ec.build_features(V, {"density": dens}, at_risk=at_risk, controls=False)
    assert list(Xr.index) == [("R1", "B"), ("R1", "C"), ("R2", "A")]
    assert list(Xr.columns) == ["density"]


# ── machine-learning prediction (optional dependency) ─────────────────────────
def _planted(seed=0, R=60, C=40):
    """Entries driven by one layer and by the region; a second layer is noise."""
    rng = np.random.default_rng(seed)
    idx, cols = [f"r{i:02d}" for i in range(R)], [f"a{j:02d}" for j in range(C)]
    signal = pd.DataFrame(rng.normal(size=(R, C)), idx, cols)
    noise = pd.DataFrame(rng.normal(size=(R, C)), idx, cols)
    region = rng.normal(size=(R, 1))
    prob = 1 / (1 + np.exp(-(1.8 * signal.values + region - 2.0)))
    target = pd.DataFrame((rng.random((R, C)) < prob).astype(float), idx, cols)
    target = target.mask(rng.random((R, C)) < 0.2)
    mat = pd.DataFrame(rng.poisson(20, (R, C)).astype(float) + 1, idx, cols)
    X = ec.build_features(mat, {"signal": signal, "noise": noise},
                          at_risk=target.notna(), controls=False)
    return X, target, signal, noise


def test_fold_assignment_keeps_groups_together():
    from econcomplex.prediction.model import _fold_of
    groups = np.repeat(np.arange(23), 7)
    fold = _fold_of(groups, 5, seed=0)
    assert all(len(set(fold[groups == g])) == 1 for g in range(23))
    assert set(fold) == set(range(5))
    assert not np.array_equal(fold, _fold_of(groups, 5, seed=1))
    with pytest.raises(ValueError):
        _fold_of(groups[:14], 5, seed=0)


def test_predict_entry_out_of_sample():
    pytest.importorskip("sklearn")
    X, target, signal, noise = _planted()
    scores = {"logit": ec.predict_entry(X, target, model="logit", seed=0),
              "gbm": ec.predict_entry(X, target, model="gbm", n_splits=2, seed=0),
              "signal": signal, "noise": noise}
    assert scores["gbm"].shape == target.shape
    assert scores["gbm"].isna().equals(target.isna())          # only cells at risk
    val = ec.validate_entry(scores, target)["value"].unstack()
    assert val.loc["logit", "auc"] > 0.8 and val.loc["gbm", "auc"] > 0.75
    assert abs(val.loc["noise", "auc"] - 0.5) < 0.05
    # the model cannot use the region effect of unseen regions: its ranking
    # of a region's options is the one of the planted layer
    assert val.loc["logit", "auc_within"] == pytest.approx(val.loc["signal", "auc_within"], abs=0.02)
    assert val.loc["logit", "brier"] < 0.25
    pd.testing.assert_frame_equal(scores["logit"], ec.predict_entry(X, target, model="logit", seed=0))
    # grouping regions into larger units, calibrated probabilities, own estimator
    from sklearn.tree import DecisionTreeClassifier
    states = {r: r[:2] for r in target.index}                  # r0, r1, ... r5
    p = ec.predict_entry(X, target, model=DecisionTreeClassifier(max_depth=3),
                         groups=states, n_splits=3, calibrate=True, seed=0)
    assert p.min().min() >= 0 and p.max().max() <= 1
    with pytest.raises(ValueError):
        ec.predict_entry(X, target, model="xgboost")
    with pytest.raises(ValueError):
        ec.predict_entry(X, target, groups={"r00": "x"})


def test_predict_entry_new_window_and_model():
    pytest.importorskip("sklearn")
    X, target, signal, _ = _planted()
    X_new, target_new, signal_new, _ = _planted(seed=9)
    probs, model = ec.predict_entry(X, target, model="logit", new_features=X_new,
                                    return_model=True)
    assert hasattr(model, "predict_proba") and probs.shape == target_new.shape
    auc = ec.validate_entry(probs, target_new)["value"]["score"]["auc"]
    assert auc > 0.8
    with pytest.raises(ValueError, match="lacks"):
        ec.predict_entry(X, target, new_features=X_new[["signal"]])


def test_predict_entry_vector_mode():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    R, C = 150, 6
    idx, cols = [f"r{i}" for i in range(R)], list("ABCDEF")
    rca0 = pd.DataFrame(rng.lognormal(0, 1, (R, C)), idx, cols)
    # entry into A happens where the region is strong in B; F never has enough entries
    target = pd.DataFrame(np.nan, idx, cols)
    target["A"] = (rca0["B"] + rng.normal(0, 0.3, R) > 1.2).astype(float)
    target["F"] = 0.0
    target.iloc[:2, 5] = 1.0
    p = ec.predict_entry(rca0, target, mode="vector", model="rf", seed=0)
    assert p.shape == target.shape and p["F"].isna().all() and p["C"].isna().all()
    auc = ec.validate_entry(p[["A"]], target[["A"]])["value"]["score"]["auc"]
    assert auc > 0.85
    new, models = ec.predict_entry(rca0, target, mode="vector", model="rf", seed=0,
                                   new_features=rca0.iloc[:10], return_model=True)
    assert new.shape == (10, C) and list(models) == ["A"]


def test_explain_entry_finds_the_planted_layer():
    pytest.importorskip("sklearn")
    X, target, _, _ = _planted()
    imp = ec.explain_entry(X, target, model="logit", n_repeats=8, seed=0)
    assert list(imp.columns) == ["importance", "std", "p_value", "direction", "significant"]
    assert imp.loc["signal", "direction"] == "+"
    assert imp.index[0] == "signal" and bool(imp.loc["signal", "significant"])
    assert imp.loc["signal", "importance"] > 0.2
    assert abs(imp.loc["noise", "importance"]) < 0.01
    assert not bool(imp.loc["noise", "significant"])


# ── input-output relatedness ──────────────────────────────────────────────────
def test_io_relatedness_known_and_blocks():
    Z = pd.DataFrame([[0.0, 6.0, 2.0], [3.0, 0.0, 1.0], [0.0, 0.0, 5.0]],
                     index=list("ABC"), columns=list("ABC"))
    r = ec.io_relatedness(Z)
    assert r.loc["A", "B"] == pytest.approx((6 / 8 + 3 / 4) / 2)
    assert r.loc["A", "C"] == pytest.approx((2 / 8 + 0 / 5) / 2)
    np.testing.assert_allclose(r.values, r.values.T)
    assert ec.io_relatedness(Z, method="max").loc["A", "C"] == pytest.approx(2 / 8)
    # purchase shares: A supplies 6 of the 6 bought by B, B supplies 3 of the 3 bought by A
    assert ec.io_relatedness(Z, shares="purchases").loc["A", "B"] == pytest.approx(1.0)
    both = ec.io_relatedness(Z, shares="both")
    assert both.loc["A", "B"] == pytest.approx(((6 / 8 + 1) / 2 + (3 / 4 + 1) / 2) / 2)
    # labels in another column order
    pd.testing.assert_frame_equal(ec.io_relatedness(Z[["C", "A", "B"]]), r)
    # two blocks of industries that do not trade with each other
    blocks = np.zeros((5, 5))
    blocks[:2, :2] = [[1, 4], [2, 1]]
    blocks[2:, 2:] = [[1, 2, 3], [4, 1, 2], [3, 3, 1]]
    out = ec.io_relatedness(blocks, shares="both", method="max")
    assert (out[:2, 2:] == 0).all() and (out[2:, :2] == 0).all() and (out[:2, :2] > 0).all()
    with pytest.raises(ValueError):
        ec.io_relatedness(np.ones((2, 3)))


# ── transition risk ───────────────────────────────────────────────────────────
def test_transition_risk_known():
    mat = pd.DataFrame([[10.0, 0.0, 30.0, 0.0], [0.0, 20.0, 0.0, 20.0], [5.0, 5.0, 5.0, 5.0]],
                       index=["R1", "R2", "R3"], columns=["coal", "oil", "wind", "soft"])
    pci = pd.Series({"soft": 2.0, "wind": 1.0, "oil": 0.0, "coal": -2.0})    # 0-1: 1, .75, .5, 0
    phi = pd.DataFrame([[0.0, 0.8, 0.1, 0.1], [0.8, 0.0, 0.4, 0.2],
                        [0.1, 0.4, 0.0, 0.6], [0.1, 0.2, 0.6, 0.0]],
                       index=mat.columns, columns=mat.columns)
    kw = {"phi": phi, "use_rca": False, "threshold": 1.0}
    out = ec.transition_risk(mat, ["coal", "oil"], pci=pci, **kw)
    np.testing.assert_allclose(out["exposure"], [0.25, 0.5, 0.5])
    np.testing.assert_allclose(out["lock_in"], [0.25 * 1.0, 0.5 * 0.5, 0.25 * 1.0 + 0.25 * 0.5])
    assert out["n_exposed"].tolist() == [1, 1, 2]
    to_coal = ((0.1 + 0.1) / 2) / (1.0 / 4)          # targets: wind and soft
    to_oil = ((0.4 + 0.2) / 2) / (1.4 / 4)
    np.testing.assert_allclose(out["outlook"], [to_coal, to_oil, (to_coal + to_oil) / 2])
    green = ec.transition_risk(mat, ["coal", "oil"], targets=["wind"], **kw)
    assert green.loc["R2", "outlook"] == pytest.approx(0.4 / (1.4 / 4))
    assert green["lock_in"].isna().all()
    # a region without exposed specializations has no outlook
    clean = mat.copy()
    clean.loc["R1", "coal"] = 0.0
    assert np.isnan(ec.transition_risk(clean, ["coal", "oil"], **kw).loc["R1", "outlook"])
    with pytest.raises(ValueError, match="not in the matrix"):
        ec.transition_risk(mat, ["gas"], **kw)


def test_explain_entry_vector_mode_and_direction():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    R, C = 200, 5
    idx, cols = [f"r{i}" for i in range(R)], list("ABCDE")
    rca0 = pd.DataFrame(rng.lognormal(0, 1, (R, C)), idx, cols)
    # entry into A is helped by B and hindered by C; D and E are irrelevant
    drive = np.log(rca0["B"]) - np.log(rca0["C"]) + rng.normal(0, 0.4, R)
    target = pd.DataFrame(np.nan, idx, cols)
    target["A"] = (drive > 0.3).astype(float)
    imp = ec.explain_entry(np.log(rca0), target, model="logit", mode="vector",
                           n_repeats=6, seed=0)
    assert list(imp.index.names) == ["target", "feature"]
    a = imp.loc["A"]
    assert set(a.index[:2]) == {"B", "C"}
    assert a.loc["B", "direction"] == "+" and a.loc["C", "direction"] == "-"
    assert bool(a.loc["B", "significant"]) and bool(a.loc["C", "significant"])
    assert not a.loc[["D", "E"], "significant"].any()
    only = ec.explain_entry(np.log(rca0), target, model="logit", mode="vector",
                            activities=["A"], n_repeats=2, seed=0)
    assert list(only.index.get_level_values("target").unique()) == ["A"]
    with pytest.raises(ValueError, match="enough entries"):
        ec.explain_entry(np.log(rca0), target, model="logit", mode="vector",
                         activities=["B"])
