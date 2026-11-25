"""Tests of the functions added in 1.2.0: flow and spatial relatedness,
density decompositions, strategy, policy lists, entry prediction, and
the corrected Complexity Outlook Gain."""

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


# ── Complexity Outlook Gain ───────────────────────────────────────────────────
def test_cog_is_the_gain_in_coi(big):
    """Adding activity c to region r raises its COI by COG_rc, net of the
    term of c itself, which stops being a missing activity."""
    m = ec.mcp(big)
    _, pci = ec.eci_pci(big)
    phi = ec.product_proximity(big)
    cog = ec.cog(big, pci, phi=phi)
    kw = {"phi": phi, "use_rca": False, "threshold": 0.5}
    coi0 = ec.coi(m, pci, **kw)
    dens0 = ec.relatedness_density(m, percent=False, **kw)
    checked = 0
    for r in m.index[:6]:
        for c in m.columns[m.loc[r] == 0][:5]:
            m2 = m.copy()
            m2.loc[r, c] = 1
            gain = ec.coi(m2, pci, **kw)[r] - coi0[r] + dens0.loc[r, c] * pci[c]
            assert cog.loc[r, c] == pytest.approx(gain, abs=1e-10)
            checked += 1
    assert checked > 10
    assert (cog.values[m.values == 1] == 0).all()


# ── skill_relatedness ─────────────────────────────────────────────────────────
def test_skill_relatedness_expected_flows_give_zero():
    flows = np.outer([1.0, 2.0, 3.0], [4.0, 1.0, 5.0])        # F_ij = a_i b_j
    np.testing.assert_allclose(
        ec.skill_relatedness(flows, exclude_diagonal=False), 0, atol=1e-12)
    np.testing.assert_allclose(
        ec.skill_relatedness(flows, exclude_diagonal=False, normalize=False), 1)


def test_skill_relatedness_known_and_symmetric():
    flows = pd.DataFrame([[0, 8, 2], [6, 0, 4], [1, 9, 0]],
                         index=list("ABC"), columns=list("ABC"), dtype=float)
    sr = ec.skill_relatedness(flows)
    ratio = 8 / (10 * 17 / 30)                                # A -> B
    assert sr.loc["A", "B"] == pytest.approx((ratio - 1) / (ratio + 1))
    assert sr.loc["A", "C"] == pytest.approx(0.0)             # 2 observed, 2 expected
    assert np.isnan(np.diag(sr)).all()
    sym = ec.skill_relatedness(flows, symmetric=True)
    np.testing.assert_allclose(sym.values, sym.values.T, equal_nan=True)
    # transform first, then average the two directions (Neffke, Otto & Weyh 2017)
    assert sym.loc["A", "B"] == pytest.approx((sr.loc["A", "B"] + sr.loc["B", "A"]) / 2)
    raw = ec.skill_relatedness(flows, normalize=False)
    raw_sym = ec.skill_relatedness(flows, normalize=False, symmetric=True)
    assert raw_sym.loc["A", "B"] == pytest.approx((raw.loc["A", "B"] + raw.loc["B", "A"]) / 2)


def test_skill_relatedness_long_table_and_empty_margin():
    long = pd.DataFrame({"o": ["A", "A", "B"], "d": ["B", "C", "C"], "n": [5.0, 5.0, 2.0]})
    sr = ec.skill_relatedness(ec.pivot_to_matrix(long, "o", "d", "n"))
    assert list(sr.index) == list(sr.columns) and set(sr.index) == {"A", "B", "C"}
    assert sr.loc["C"].isna().all()          # C has no outflow: expected flow is 0
    assert sr.loc["B", "A"] != sr.loc["B", "A"]   # A has no inflow either
    with pytest.raises(ValueError, match="square"):
        ec.skill_relatedness(np.ones((2, 3)))


# ── spatial_density ───────────────────────────────────────────────────────────
def test_spatial_density_known(V):
    W = pd.DataFrame([[0, 1, 1], [1, 0, 0], [0, 0, 0]], index=V.index, columns=V.index)
    sd = ec.spatial_density(V, W)
    np.testing.assert_allclose(sd.loc["R1"], [0.5, 1.0, 1.0])   # neighbours R2, R3
    np.testing.assert_allclose(sd.loc["R2"], [1.0, 0.0, 0.0])   # only neighbour: R1
    assert sd.loc["R3"].isna().all()                            # no neighbours
    # the diagonal is ignored and labels are matched in any order
    W2 = W + np.eye(3) * 7
    shuffled = W2.loc[["R3", "R1", "R2"], ["R2", "R3", "R1"]]
    pd.testing.assert_frame_equal(ec.spatial_density(V, shuffled), sd)
    with pytest.raises(ValueError):
        ec.spatial_density(V, -W)


# ── density decompositions ────────────────────────────────────────────────────
def test_density_contributions_sum_to_density(big):
    phi = ec.product_proximity(big)
    dens = ec.relatedness_density(big, phi=phi)
    m = ec.mcp(big)
    for r, c in [(big.index[0], big.columns[3]), (big.index[7], big.columns[0]),
                 (big.index[-1], big.columns[-1])]:
        contrib = ec.density_contributions(big, r, c, phi=phi)
        assert contrib.sum() == pytest.approx(dens.loc[r, c])
        assert (m.loc[r, contrib.index] == 1).all()
        assert contrib.is_monotonic_decreasing
    share = ec.density_contributions(big, big.index[0], big.columns[3], phi=phi, percent=False)
    assert share.sum() == pytest.approx(dens.iloc[0, 3] / 100)
    # arrays: positions instead of labels
    by_pos = ec.density_contributions(big.values, 0, 3, phi=phi.values)
    assert by_pos.sum() == pytest.approx(dens.iloc[0, 3])


def test_decomposed_density_additive_and_incomplete():
    additive = np.add.outer([1.0, 2.0, 5.0], [0.0, 10.0, -3.0, 4.0])
    np.testing.assert_allclose(ec.decomposed_density(additive), 0, atol=1e-12)

    rng = np.random.default_rng(0)
    d = rng.random((8, 6))
    full = ec.decomposed_density(pd.DataFrame(d))
    np.testing.assert_allclose(
        full.values, d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean())

    # incomplete matrix: must equal the OLS residual on region and activity dummies
    d[rng.random(d.shape) < 0.3] = np.nan
    res = ec.decomposed_density(d)
    rr, cc = np.nonzero(~np.isnan(d))
    X = np.zeros((rr.size, 8 + 6))
    X[np.arange(rr.size), rr] = 1
    X[np.arange(rr.size), 8 + cc] = 1
    y = d[rr, cc]
    ols = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(res[rr, cc], ols, atol=1e-8)
    assert np.isnan(res[np.isnan(d)]).all()


# ── strategy ──────────────────────────────────────────────────────────────────
def test_desirability_index_known():
    ind = pd.DataFrame({"pci": [1.0, 0.0, -1.0], "wage": [10.0, 30.0, 20.0]},
                       index=["A", "B", "C"])
    z = (ind - ind.mean()) / ind.std(ddof=0)
    np.testing.assert_allclose(ec.desirability_index(ind), z.mean(axis=1))
    weighted = ec.desirability_index(ind, weights={"wage": 1, "pci": 3})   # by name
    np.testing.assert_allclose(weighted, (3 * z["pci"] + z["wage"]) / 4)
    np.testing.assert_allclose(ec.desirability_index(ind, weights=[1, -1]),
                               (z["pci"] - z["wage"]) / 2)
    raw = ec.desirability_index(ind, standardize=None)
    np.testing.assert_allclose(raw, ind.mean(axis=1))
    np.testing.assert_allclose(ec.desirability_index(ind, standardize="minmax"),
                               [0.5, 0.75, 0.25])
    # a constant indicator adds nothing; a missing one gives NaN
    ind["flat"] = 5.0
    np.testing.assert_allclose(ec.desirability_index(ind), z.mean(axis=1) * 2 / 3)
    ind.loc["A", "wage"] = np.nan
    assert np.isnan(ec.desirability_index(ind)["A"])
    with pytest.raises(ValueError):
        ec.desirability_index(ind, weights=[0, 0, 0])


def test_weight_sensitivity():
    rng = np.random.default_rng(1)
    ind = pd.DataFrame(rng.normal(size=(12, 3)), columns=["a", "b", "c"],
                       index=[f"act{i}" for i in range(12)])
    ind.loc["act0"] = ind.max() + 1                       # best on every indicator
    ind.loc["act1"] = ind.min() - 1                       # worst on every indicator
    out = ec.weight_sensitivity(ind, n_draws=300, top=3, seed=0)
    assert out.index[0] == "act0" and out.index[-1] == "act1"
    assert out.loc["act0", ["rank_mean", "rank_p95", "top_share"]].tolist() == [1, 1, 1]
    assert out.loc["act1", "rank_p05"] == 12 and out.loc["act1", "top_share"] == 0
    np.testing.assert_allclose(out["score"].sort_index(),
                               ec.desirability_index(ind).sort_index())
    pd.testing.assert_frame_equal(out, ec.weight_sensitivity(ind, n_draws=300, top=3, seed=0))
    # a tight concentration barely moves the ranks
    tight = ec.weight_sensitivity(ind, n_draws=300, concentration=1e4, seed=0)
    assert (tight["rank_p95"] - tight["rank_p05"]).max() <= 1


def test_strategy_quadrants():
    feas = pd.DataFrame([[10.0, 40.0, 30.0, 20.0], [np.nan, 5.0, 1.0, 9.0]],
                        index=["R1", "R2"], columns=list("ABCD"))
    pci = pd.Series({"D": -2.0, "A": 2.0, "C": -1.0, "B": 1.5})     # any order
    q = ec.strategy_quadrants(feas, pci)
    assert q.loc["R1"].tolist() == ["long bet", "short jump", "low value", "avoid"]
    # R2: A is not an option; cuts over B, C, D -> feasibility 5, desirability -1
    assert pd.isna(q.loc["R2", "A"])
    assert q.loc["R2"].tolist()[1:] == ["long bet", "avoid", "low value"]
    fixed = ec.strategy_quadrants(feas, pci, cut=(8.0, 0.0))
    assert fixed.loc["R2"].tolist()[1:] == ["long bet", "avoid", "low value"]
    assert fixed.loc["R1", "A"] == "short jump"
    pooled = ec.strategy_quadrants(feas.values, pci[feas.columns].values, by_region=False)
    assert pooled[1, 3] == "avoid" and pooled[0, 1] == "short jump"
    # region-specific desirability (R x C)
    des = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                       index=feas.index, columns=feas.columns)
    assert ec.strategy_quadrants(feas, des).loc["R1", "D"] == "long bet"


# ── policy lists ──────────────────────────────────────────────────────────────
def test_list_complexity_potential_reproduces_coi(big):
    _, pci = ec.eci_pci(big)
    phi = ec.product_proximity(big)
    dens = ec.relatedness_density(big, phi=phi, percent=False)
    lists = {"all": list(big.columns), "first": list(big.columns[:4])}
    out = ec.list_complexity_potential(big, pci, dens, lists)
    pot = out["potential"].unstack()
    np.testing.assert_allclose(pot["all"] * big.shape[1], ec.coi(big, pci, phi=phi))
    m = ec.mcp(big)
    first = big.columns[:4]
    np.testing.assert_allclose(out["presence"].unstack()["first"], (m[first] * pci[first]).sum(axis=1))
    np.testing.assert_allclose(pot["first"], ((1 - m[first]) * dens[first] * pci[first]).sum(axis=1) / 4)
    assert (out["n_list"].unstack()["first"] == 4).all()
    np.testing.assert_array_equal(out["n_present"].unstack()["first"], m[first].sum(axis=1))
    # Mealy & Teytelboym's GCP: mean over the list activities the region lacks
    gcp = ec.list_complexity_potential(big, pci, dens, lists, average="missing")
    n_missing = (1 - m[first]).sum(axis=1)
    expected = (((1 - m[first]) * dens[first] * pci[first]).sum(axis=1) / n_missing)
    np.testing.assert_allclose(gcp["potential"].unstack()["first"], expected.where(n_missing > 0))
    pd.testing.assert_series_equal(gcp["presence"], out["presence"])
    # same lists as a boolean table
    table = pd.DataFrame({k: big.columns.isin(v) for k, v in lists.items()}, index=big.columns)
    pd.testing.assert_frame_equal(ec.list_complexity_potential(big, pci, dens, table), out)
    with pytest.raises(ValueError, match="not in the matrix"):
        ec.list_complexity_potential(big, pci, dens, {"x": ["nope"]})


def test_list_density_parts_add_up(big):
    phi = ec.product_proximity(big)
    cols = list(big.columns)
    parts = ec.list_density(big, {"a": cols[:5], "b": cols[5:9], "c": cols[9:]}, phi=phi)
    total = ec.relatedness_density(big, phi=phi)
    pd.testing.assert_frame_equal(parts["a"] + parts["b"] + parts["c"], total)
    assert (parts["a"].values >= 0).all()
    arr = ec.list_density(big.values, {"a": [0, 1, 2, 3, 4]}, phi=phi.values, percent=False)
    np.testing.assert_allclose(arr["a"], parts["a"].values / 100)


def test_missing_links(V):
    # M = [[1,0,0],[0,1,1],[1,1,1]]
    feas = pd.DataFrame([[0.0, 0.6, 0.1], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        index=V.index, columns=V.columns)
    links = {"upstream": ["A"], "core": ["B"], "downstream": ["C"]}
    out = ec.missing_links(V, links, feas, cut=0.5)
    assert out.loc["R1"].tolist() == ["present", "feasible", "missing"]
    assert out.loc["R2"].tolist() == ["missing", "present", "present"]
    assert out.loc["R3"].tolist() == ["present"] * 3
    two = ec.missing_links(V, {("x", "up"): ["A", "B"], ("x", "down"): ["C"]}, feas, cut=0.05)
    assert two.loc["R1", ("x", "down")] == "feasible"


# ── entry prediction ──────────────────────────────────────────────────────────
def test_entry_targets_known():
    r0 = pd.DataFrame([[0.2, 1.4, 0.0, 0.7], [0.0, 0.1, 0.3, 0.4]],
                      index=["R1", "R2"], columns=list("ABCD"))
    r1 = pd.DataFrame([[1.3, 1.1, 0.4, 2.0], [1.0, 0.9, 5.0, 0.0]],
                      index=["R1", "R2"], columns=list("ABCD"))
    target, at_risk = ec.entry_targets(r0, r1)
    np.testing.assert_array_equal(target.values, [[1, np.nan, 0, np.nan], [1, 0, 1, 0]])
    np.testing.assert_array_equal(at_risk.values, [[1, 0, 1, 0], [1, 1, 1, 1]])
    # minimum size, t1 in another order and without activity D
    vals = pd.DataFrame([[50.0, 9.0, 9.0], [3.0, 9.0, 80.0]], index=["R1", "R2"], columns=list("ABC"))
    t2, _ = ec.entry_targets(r0, r1.loc[["R2", "R1"], list("CBA")], values_t1=vals, min_value=10)
    np.testing.assert_array_equal(t2.values, [[1, np.nan, 0, np.nan], [0, 0, 1, 0]])
    arr, _ = ec.entry_targets(r0.values, r1.values, lower=1.0)
    assert arr[0, 3] == 1 and np.isnan(arr[0, 1])
    with pytest.raises(ValueError):
        ec.entry_targets(r0, r1, min_value=10)


def test_validate_entry_known_values():
    target = np.array([[0.0, 0.0, 1.0, 1.0]])
    score = np.array([[0.1, 0.4, 0.35, 0.8]])
    v = ec.validate_entry(score, target, k=(1, 2))["value"]["score"]
    assert v["auc"] == pytest.approx(0.75)
    assert v["auc_within"] == pytest.approx(0.75)
    assert v["average_precision"] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    assert v["precision@1"] == 1 and v["recall@1"] == 0.5
    assert v["precision@2"] == 0.5 and v["recall@2"] == 0.5
    assert v["brier"] == pytest.approx(np.mean((score - target) ** 2))
    # not a probability -> no Brier score; cells outside the risk set are ignored
    t2 = np.array([[0.0, 0.0, 1.0, 1.0, np.nan]])
    v2 = ec.validate_entry(np.array([[1.0, 4.0, 3.5, 8.0, 99.0]]), t2, k=(1,))["value"]["score"]
    assert v2["auc"] == pytest.approx(0.75) and np.isnan(v2["brier"])


def _planted(seed=0, R=40, C=60):
    rng = np.random.default_rng(seed)
    idx, cols = [f"r{i}" for i in range(R)], [f"a{j}" for j in range(C)]
    signal = pd.DataFrame(rng.normal(size=(R, C)), idx, cols)
    region = rng.normal(size=(R, 1)) * 1.5            # some regions enter more
    prob = 1 / (1 + np.exp(-(1.5 * signal.values + region - 2.5)))
    target = pd.DataFrame((rng.random((R, C)) < prob).astype(float), idx, cols)
    target = target.mask(rng.random((R, C)) < 0.2)    # not at risk
    scores = {"signal": signal,
              "noise": pd.DataFrame(rng.normal(size=(R, C)), idx, cols),
              "region": pd.DataFrame(np.repeat(region, C, axis=1), idx, cols)}
    return scores, target


def test_validate_entry_planted_signal():
    scores, target = _planted()
    val = ec.validate_entry(scores, target)["value"].unstack()
    assert val.loc["signal", "auc"] > 0.75 and val.loc["signal", "auc_within"] > 0.75
    assert abs(val.loc["noise", "auc"] - 0.5) < 0.05
    assert abs(val.loc["noise", "auc_within"] - 0.5) < 0.05
    # a region-level score ranks regions but not the options of a region
    assert val.loc["region", "auc"] > 0.6
    assert val.loc["region", "auc_within"] == pytest.approx(0.5)
    assert val.loc["signal", "precision@10"] > 2 * val.loc["noise", "precision@10"]
    assert val.loc["signal", "recall@20"] > val.loc["signal", "recall@10"]
    # label alignment
    shuffled = scores["signal"].sample(frac=1, random_state=1).sample(frac=1, axis=1, random_state=2)
    again = ec.validate_entry(shuffled, target)["value"]["score"]
    assert again["auc"] == pytest.approx(val.loc["signal", "auc"])


def test_validate_entry_bootstrap_and_baseline():
    scores, target = _planted()
    out = ec.validate_entry(scores, target, k=(10,), baseline="noise", n_boot=60, seed=0)
    assert list(out.columns) == ["value", "lo", "hi", "diff", "diff_lo", "diff_hi"]
    sig = out.loc["signal"]
    inside = sig.dropna()
    assert ((inside["lo"] <= inside["value"]) & (inside["value"] <= inside["hi"])).all()
    assert sig.loc["auc", "diff_lo"] > 0 and sig.loc["precision@10", "diff_lo"] > 0
    assert (out.loc["noise", "diff"].dropna() == 0).all()   # Brier: not a probability
    assert sig.loc["auc", "diff"] == pytest.approx(sig.loc["auc", "value"] - out.loc[("noise", "auc"), "value"])
    pd.testing.assert_frame_equal(
        out, ec.validate_entry(scores, target, k=(10,), baseline="noise", n_boot=60, seed=0))
    with pytest.raises(ValueError):
        ec.validate_entry(scores, target, baseline="missing")


def test_validate_entry_matches_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = (rng.random(400) < 0.2).astype(float)
    s = np.round(rng.random(400) * 0.6 + 0.3 * y, 1)          # many ties
    v = ec.validate_entry(s.reshape(1, -1), y.reshape(1, -1))["value"]["score"]
    assert v["auc"] == pytest.approx(metrics.roc_auc_score(y, s))
    assert v["average_precision"] == pytest.approx(metrics.average_precision_score(y, s))
    assert v["brier"] == pytest.approx(metrics.brier_score_loss(y, s))
