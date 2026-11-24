"""Quality tests: 1.0.1 baseline, label alignment, input validation, and
known-answer checks for functions that had no direct test."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import econcomplex as ec

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def V():
    """3 regions x 3 activities with round RCA values.

    RCA = [[2, 0, 0], [0, 2, 2], [1, 1, 1]]  ->  M = [[1,0,0],[0,1,1],[1,1,1]]
    """
    return pd.DataFrame(
        [[4.0, 0.0, 0.0], [0.0, 2.0, 2.0], [4.0, 2.0, 2.0]],
        index=["R1", "R2", "R3"], columns=["A", "B", "C"],
    )


@pytest.fixture
def B():
    """4 events x 3 technologies, binary incidence."""
    return pd.DataFrame(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1]],
        index=["p1", "p2", "p3", "p4"], columns=["t1", "t2", "t3"], dtype=float,
    )


@pytest.fixture
def big():
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=42)
    return ec.pivot_to_matrix(df, "loc", "act", "val")


def shuffled(obj, seed=3):
    return obj.sample(frac=1, random_state=seed)


# ── Baseline: results of 1.0.1 must be reproduced ─────────────────────────────
def test_baseline_1_0_1_reproduced():
    base = np.load(os.path.join(HERE, "data", "baseline_1_0_1.npz"))
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=42)
    mat = ec.pivot_to_matrix(df, "loc", "act", "val")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in ("eigenvector", "reflections", "fitness"):
            e, p = ec.eci_pci(mat, method=m)
            np.testing.assert_allclose(e.values, base[f"eci_{m}"], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(p.values, base[f"pci_{m}"], rtol=1e-7, atol=1e-9)
        res = ec.compute_complexity(
            df, cols={"loc": "loc", "act": "act", "val": "val"}
        ).sort_values(["loc", "act"])
    for col in ["rca", "mcp", "diversity", "ubiquity", "eci", "pci",
                "density", "distance", "coi", "cog"]:
        np.testing.assert_allclose(res[col].to_numpy(dtype=float),
                                   base[f"pipe_{col}"], rtol=1e-7, atol=1e-9,
                                   err_msg=col)


# ── Label alignment: same labels in another order give the same result ────────
def test_vector_inputs_are_aligned_by_label(big):
    rng = np.random.default_rng(0)
    by_loc = pd.Series(rng.uniform(1e3, 5e4, len(big)), index=big.index)
    for fn in (ec.prody, ec.expy, ec.pgi, ec.peii, ec.rpop):
        a, b = fn(big, by_loc), fn(big, shuffled(by_loc))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), err_msg=fn.__name__)
    for fn in (ec.hoover_index, ec.hoover_gini):
        a, b = fn(big, pop=by_loc), fn(big, pop=shuffled(by_loc))
        np.testing.assert_allclose(a.values, b.values, err_msg=fn.__name__)


def test_pci_and_phi_are_aligned_by_label(big):
    _, pci = ec.eci_pci(big)
    phi = ec.proximity(big, compute="product")["product"]
    cols = list(phi.columns[::-1])
    phi_r = phi.loc[cols, cols]
    np.testing.assert_allclose(ec.coi(big, pci, phi=phi),
                               ec.coi(big, shuffled(pci), phi=phi_r))
    np.testing.assert_allclose(ec.cog(big, pci, phi=phi).values,
                               ec.cog(big, shuffled(pci), phi=phi_r).values)
    for fn in (ec.density, ec.distance, ec.relatedness_density_external):
        np.testing.assert_allclose(fn(big, phi=phi).values, fn(big, phi=phi_r).values,
                                   equal_nan=True, err_msg=fn.__name__)


def test_subnational_eci_aligns_and_accepts_superset(V):
    pci = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    eci, _ = ec.subnational_eci(V, pci, standardize=False)
    np.testing.assert_allclose(eci.values, [1.0, 2.5, 2.0])
    wider = pd.Series({"Z": 9.0, "C": 3.0, "A": 1.0, "B": 2.0})
    eci2, _ = ec.subnational_eci(V, wider, standardize=False)
    np.testing.assert_allclose(eci2.values, [1.0, 2.5, 2.0])


def test_cross_relatedness_aligns_rows_of_x_phi(big):
    other = pd.DataFrame(np.random.default_rng(1).poisson(8, (len(big), 4)),
                         index=big.index, columns=list("wxyz"), dtype=float)
    x_phi = ec.cross_proximity(big, other)
    a = ec.cross_relatedness(big, x_phi)
    b = ec.cross_relatedness(big, shuffled(x_phi))
    np.testing.assert_allclose(a.values, b.values)


def test_eor_and_active_are_aligned_by_label(B, big):
    eor = ec.ease_of_recombination(B)
    np.testing.assert_allclose(ec.modular_complexity(B, eor=eor),
                               ec.modular_complexity(B, eor=shuffled(eor)))
    adj = ec.proximity_network(big, phi_threshold=0.4)
    active = ec.mcp(big).iloc[0]
    p1 = ec.activation_probabilities(adj, active, B=0.5, alpha=1.0)
    p2 = ec.activation_probabilities(adj, shuffled(active), B=0.5, alpha=1.0)
    np.testing.assert_allclose(p1.values, p2.values)


def test_missing_label_raises_and_plain_arrays_stay_positional(V):
    with pytest.raises(ValueError, match="missing"):
        ec.prody(V, pd.Series({"R1": 1.0, "R2": 2.0}))        # R3 absent
    with pytest.raises(ValueError):
        ec.prody(V, [1.0, 2.0])                                # wrong length
    by_pos = ec.prody(V, [10.0, 20.0, 30.0])                   # list: positional
    by_default_index = ec.prody(V, pd.Series([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(by_pos.values, [50 / 3, 70 / 3, 70 / 3])
    np.testing.assert_allclose(by_default_index.values, by_pos.values)


# ── Input validation ──────────────────────────────────────────────────────────
def test_nan_and_negative_values_raise_clear_errors(V):
    bad = V.copy()
    bad.iloc[0, 0] = np.nan
    for fn in (ec.rca, ec.eci_pci, ec.proximity, ec.herfindahl):
        with pytest.raises(ValueError, match="NaN"):
            fn(bad)
    neg = V.copy()
    neg.iloc[0, 0] = -1.0
    for fn in (ec.rca, ec.eci_pci, ec.herfindahl):
        with pytest.raises(ValueError, match="negative"):
            fn(neg)


def test_binarize_still_accepts_any_real_matrix():
    out = ec.binarize(np.array([[-1.0, 0.5], [1.0, 2.0]]), threshold=1.0)
    np.testing.assert_array_equal(out, [[0, 0], [1, 1]])


# ── eci_optimization: time limit keeps a feasible portfolio ───────────────────
def test_eci_optimization_time_limit_returns_feasible_portfolios():
    rng = np.random.default_rng(0)
    m0 = ec.pivot_to_matrix(ec.make_sample_data(n_locs=30, n_acts=20, seed=0),
                            "loc", "act", "val")
    mats = {2000: m0}
    for a, b in [(2000, 2005), (2005, 2010)]:
        mats[b] = mats[a] * rng.lognormal(0, 0.3, mats[a].shape) * 1.1
    panel = pd.concat([ec.melt_matrix(m, "loc", "act", "val").assign(year=y)
                       for y, m in mats.items()])
    panel = panel[panel["val"] > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ec.calibrate_steppingstone(panel, "loc", "act", "val", "year",
                                           horizon=10, steppingstone=5)
        exact = ec.eci_optimization(mats[2010], model, delta_eci=0.1)
        rushed = ec.eci_optimization(mats[2010], model, delta_eci=0.1,
                                     time_limit=1e-9)
    assert "optimal" in exact.columns and exact["optimal"].all()
    assert len(rushed) > 0
    assert (rushed["eci_achieved"] >= rushed["eci_target"] - 1e-9).all()
    assert set(rushed["location"]) == set(exact["location"])


# ── Known answers for functions that had no direct test ───────────────────────
def test_rpop_known(V):
    out = ec.rpop(V, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(out.values, [[1, 0, 0], [0, 2, 2], [2, 2, 2]])


def test_normalized_ubiquity_known(V):
    np.testing.assert_allclose(ec.normalized_ubiquity(V).values, [1.5, 0.75, 0.75])


def test_binarize_and_normalizers_known():
    np.testing.assert_array_equal(ec.binarize(np.array([[0.5, 1.0], [2.0, 0.0]])),
                                  [[0, 1], [1, 0]])
    np.testing.assert_allclose(ec.normalize_zscore(np.array([1.0, 2.0, 3.0])),
                               [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    np.testing.assert_array_equal(ec.normalize_zscore(np.array([4.0, 4.0])), [0, 0])
    np.testing.assert_allclose(ec.normalize_01(np.array([2.0, 4.0, 6.0])), [0, 0.5, 1])
    np.testing.assert_array_equal(ec.normalize_01(np.array([4.0, 4.0])), [0, 0])


def test_subnational_eci_zero_diversity_is_nan():
    mat = pd.DataFrame([[4.0, 0.0], [0.0, 4.0], [0.0, 0.0]],
                       index=["R1", "R2", "R3"], columns=["A", "B"])
    eci, _ = ec.subnational_eci(mat, pd.Series({"A": 1.0, "B": 3.0}),
                                standardize=False)
    np.testing.assert_allclose(eci.values, [1.0, 3.0, np.nan])
    z, _ = ec.subnational_eci(mat, pd.Series({"A": 1.0, "B": 3.0}))
    np.testing.assert_allclose(z.values, [-1.0, 1.0, np.nan])


def test_density_internal_external_known(V):
    phi = ec.proximity(V, compute="product")["product"]
    np.testing.assert_allclose(phi.values, [[0, .5, .5], [.5, 0, 1], [.5, 1, 0]])
    third = 100 / 3
    np.testing.assert_allclose(ec.relatedness_density(V, phi=phi).values,
                               [[0, third, third], [100, 2 * third, 2 * third],
                                [100, 100, 100]])
    nan = np.nan
    np.testing.assert_allclose(ec.relatedness_density_internal(V, phi=phi).values,
                               [[0, nan, nan], [nan, 2 * third, 2 * third],
                                [100, 100, 100]])
    np.testing.assert_allclose(ec.relatedness_density_external(V, phi=phi).values,
                               [[nan, third, third], [100, nan, nan],
                                [nan, nan, nan]])


def test_relative_relatedness_is_zscore_over_option_set(big):
    rel = ec.relative_relatedness(big).values
    held = ec.mcp(big).values == 1
    assert np.isnan(rel[held]).all()
    checked = 0
    for i in range(rel.shape[0]):
        vals = rel[i, ~held[i]]
        if np.isfinite(vals).all() and len(vals) > 1:
            assert abs(vals.mean()) < 1e-9 and abs(vals.std() - 1) < 1e-9
            checked += 1
    assert checked > 0


def test_cooccurrence_family_known(B):
    np.testing.assert_allclose(ec.co_occurrence(B).values,
                               [[0, 2, 1], [2, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(np.diag(ec.co_occurrence(B, diagonal=True)), [3, 2, 2])
    jac = ec.relatedness_index(B, method="jaccard").values
    np.testing.assert_allclose([jac[0, 1], jac[0, 2], jac[1, 2]], [2 / 3, 1 / 4, 1 / 3])
    assert np.isclose(ec.relatedness_index(B, method="cosine").values[0, 1],
                      2 / np.sqrt(6))
    assert np.isclose(ec.relatedness_index(B, method="association").values[0, 1], 4 / 3)
    assert np.isclose(ec.relatedness_index(B, method="probability").values[0, 1], 4 / 9)
    z = ec.z_score_novelty(B).values
    assert np.isclose(z[0, 1], 1.0) and np.isclose(z[1, 0], 1.0)
    np.testing.assert_allclose(np.diag(z), 0)


def test_location_quotients_known(V):
    np.testing.assert_allclose(ec.location_quotient(V).values,
                               [[2, 0, 0], [0, 2, 2], [1, 1, 1]])
    np.testing.assert_allclose(ec.location_quotient_avg(V).values, [2, 2, 1])
    np.testing.assert_allclose(ec.hachman_index(V).values, [0.5, 0.5, 1])


def test_export_similarity_known(V):
    two = ec.export_similarity(V.iloc[:2])      # opposite specializations
    assert np.isclose(two.loc["R1", "R2"], -1.0)
    twin = pd.concat([V.iloc[:2], V.iloc[[0]].rename(index={"R1": "R1b"})])
    sim = ec.export_similarity(twin)
    assert np.isclose(sim.loc["R1", "R1b"], 1.0)
    np.testing.assert_allclose(np.diag(sim), 1.0)


def test_hoover_index_known(V):
    np.testing.assert_allclose(ec.hoover_index(V).values, [25, 25, 25])


def test_modular_complexity_known(B):
    np.testing.assert_allclose(ec.ease_of_recombination(B).values, [2 / 3, 1, 1 / 2])
    mc = ec.modular_complexity(B).values
    np.testing.assert_allclose(mc, [1.2, 1.5, 18 / 13, 2.0])
    np.testing.assert_allclose(
        ec.modular_complexity_avg(B).values,
        [(1.2 + 1.5 + 18 / 13) / 3, (1.2 + 18 / 13) / 2, (18 / 13 + 2) / 2])


def test_growth_matrix_known(V):
    np.testing.assert_allclose(ec.growth_matrix(V, V * 1.5).values,
                               [[50, 0, 0], [0, 50, 50], [50, 50, 50]])


def test_exit_and_summary_known():
    b1 = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["R1", "R2"], columns=["A", "B"])
    b2 = pd.DataFrame([[0.0, 1.0], [0.0, 1.0]], index=["R1", "R2"], columns=["A", "B"])
    kw = dict(use_rca=False, threshold=1.0)
    np.testing.assert_array_equal(ec.exit([b1, b2], **kw).values, [[1, 0], [0, 0]])
    np.testing.assert_array_equal(ec.entry([b1, b2], **kw).values, [[0, 1], [0, 0]])
    s = ec.entry_exit_summary([b1, b2], **kw).set_index(["location", "activity"])
    assert s.loc[("R1", "A"), "n_exits"] == 1 and s.loc[("R1", "A"), "net_change"] == -1
    assert s.loc[("R1", "B"), "n_entries"] == 1 and s.loc[("R1", "B"), "net_change"] == 1
    assert len(s) == 2


def test_coi_cog_known(V):
    pci = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    phi = ec.proximity(V, compute="product")["product"]
    np.testing.assert_allclose(ec.complexity_outlook_index(V, pci, phi=phi).values,
                               [5 / 3, 1.0, 0.0])
    np.testing.assert_allclose(ec.complexity_outlook_gain(V, pci, phi=phi).values,
                               [[0, 2, 4 / 3], [0, 0, 0], [0, 0, 0]])


def test_pgi_peii_known(V):
    by_region = pd.Series({"R1": 0.2, "R2": 0.4, "R3": 0.6})
    expected = [1 / 3, 7 / 15, 7 / 15]
    np.testing.assert_allclose(ec.product_gini_index(V, by_region).values, expected)
    np.testing.assert_allclose(ec.product_emissions_index(V, by_region).values, expected)


# ── API consistency: shortcuts, scale option, builtin-safe alias ──────────────
def test_proximity_shortcuts_return_the_matrix(big):
    both = ec.proximity(big)
    pd.testing.assert_frame_equal(ec.product_proximity(big), both["product"])
    pd.testing.assert_frame_equal(ec.location_proximity(big), both["location"])
    pd.testing.assert_frame_equal(
        ec.product_proximity(big, continuous=True, continuous_method="cosine"),
        ec.cosine_proximity(big))


def test_density_share_scale_matches_distance(big):
    share = ec.relatedness_density(big, percent=False)
    np.testing.assert_allclose(share.values * 100, ec.relatedness_density(big).values)
    np.testing.assert_allclose(1 - share.values, ec.distance(big).values)
    ext = ec.relatedness_density_external(big, percent=False).values
    assert np.nanmax(ext) <= 1.0


def test_exits_is_exit():
    assert ec.exits is ec.exit
