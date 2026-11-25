"""Tests of the symmetric RCA and of the continuous (full) relatedness
density of Cicerone, McCann & Venhorst (2026)."""

import numpy as np
import pandas as pd
import pytest

import econcomplex as ec


@pytest.fixture
def big():
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=42)
    return ec.pivot_to_matrix(df, "loc", "act", "val")


def test_rsca_dalum_and_iapadre(big):
    r = ec.rca(big)
    np.testing.assert_allclose(ec.rsca(big), (r - 1) / (r + 1))

    # Iapadre (2001) as printed in Cicerone et al. (2026), eqs. (5)-(7):
    # the region against the other regions, the activity against the others
    x = big.to_numpy()
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_r = x[i].sum()                                  # region, all activities
            x_R = x.sum() - x_r                               # other regions
            z_R = x[:, j].sum() - x[i, j]                     # activity j elsewhere
            rca_z = (x[i, j] / x_r) / (z_R / x_R)
            rca_Z = ((x_r - x[i, j]) / x_r) / ((x_R - z_R) / x_R)
            out[i, j] = (rca_z - rca_Z) / (rca_z + rca_Z)
    iap = ec.rsca(big, method="iapadre")
    np.testing.assert_allclose(iap, out)

    # both are symmetric around the same cut as the Balassa index
    assert ((iap > 0) == (r > 1)).all().all()
    assert ((ec.rsca(big) > 0) == (r > 1)).all().all()
    assert iap.values.min() >= -1 and iap.values.max() <= 1
    assert (iap.values[big.values == 0] == -1).all()
    assert ec.rsca(big.values, "iapadre").shape == big.shape
    with pytest.raises(ValueError):
        ec.rsca(big, method="laursen")


def test_rsca_degenerate_cells():
    # activity B exists only in R1; activity C nowhere; R3 has a single activity
    mat = pd.DataFrame([[2.0, 3.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                       index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    iap = ec.rsca(mat, method="iapadre")
    assert iap.loc["R1", "B"] == 1.0            # nobody else has it
    assert (iap["C"] == -1.0).all()             # nobody has it
    assert iap.loc["R2", "A"] == 1.0            # all of R2 is in A, others are not
    assert np.isfinite(iap.values).all()


def test_full_relatedness_by_hand():
    mat = pd.DataFrame([[6.0, 2.0, 0.0], [1.0, 5.0, 4.0], [3.0, 3.0, 6.0]],
                       index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    phi = pd.DataFrame([[9.0, 0.5, 0.2], [0.5, 9.0, 0.8], [0.2, 0.8, 9.0]],
                       index=mat.columns, columns=mat.columns)     # diagonal must be ignored
    w = ec.rsca(mat, method="iapadre") + 1
    full = ec.full_relatedness(mat, phi=phi)
    # around A: the other activities are B (0.5) and C (0.2)
    expected = (0.5 * w["B"] + 0.2 * w["C"]) / 0.7
    np.testing.assert_allclose(full["A"], expected)
    expected_c = (0.2 * w["A"] + 0.8 * w["B"]) / 1.0
    np.testing.assert_allclose(full["C"], expected_c)
    # phi in another order gives the same result
    cols = ["C", "A", "B"]
    pd.testing.assert_frame_equal(ec.full_relatedness(mat, phi=phi.loc[cols, cols]), full)


@pytest.mark.parametrize("weights", ["iapadre", "dalum", "capped"])
def test_full_relatedness_parts_add_up(big, weights):
    phi = ec.product_proximity(big)
    kw = {"phi": phi, "weights": weights}
    full = ec.full_relatedness(big, **kw)
    spec = ec.full_relatedness(big, part="specialized", **kw)
    rest = ec.full_relatedness(big, part="non_specialized", **kw)
    np.testing.assert_allclose(spec + rest, full, atol=1e-12)
    assert (spec.values >= 0).all() and (rest.values >= 0).all()
    assert full.values.max() <= (1 if weights == "capped" else 2) + 1e-12
    # the non-specialized part is what the standard density throws away
    assert rest.values.sum() > 0


def test_full_relatedness_capped_nests_the_standard_density(big):
    phi = ec.product_proximity(big)                 # zero diagonal
    standard = ec.relatedness_density(big, phi=phi, percent=False)
    spec = ec.full_relatedness(big, phi=phi, weights="capped", part="specialized")
    np.testing.assert_allclose(spec, standard)
    full = ec.full_relatedness(big, phi=phi, weights="capped")
    assert (full.values >= standard.values - 1e-12).all()
    # arrays and the internally computed proximity
    np.testing.assert_allclose(ec.full_relatedness(big.values, weights="capped"), full)
    with pytest.raises(ValueError):
        ec.full_relatedness(big, part="unspecialized")
    with pytest.raises(ValueError):
        ec.full_relatedness(big, weights="binary")


# ── PCA weights (Romero et al. 2025) ──────────────────────────────────────────
def test_pca_weights():
    rng = np.random.default_rng(0)
    common = rng.normal(size=300)
    ind = pd.DataFrame({"a": common + 0.1 * rng.normal(size=300),
                        "b": common + 0.1 * rng.normal(size=300),
                        "noise": rng.normal(size=300)})
    w = ec.pca_weights(ind, variance=0.6)
    assert w.sum() == pytest.approx(1) and (w >= 0).all()
    assert w["a"] == pytest.approx(w["b"], abs=0.02) and w["noise"] < 0.05
    # one component: squared loadings of the first eigenvector
    vec = np.linalg.eigh(np.corrcoef(ind.values, rowvar=False))[1][:, -1]
    np.testing.assert_allclose(w.values, vec ** 2)
    # all components kept: equal weights
    np.testing.assert_allclose(ec.pca_weights(ind, variance=1.0), 1 / 3)
    # dimensions get equal weight; a single-indicator dimension keeps all of its share
    g = ec.pca_weights(ind, groups={"x": ["a", "b"], "y": ["noise"]})
    assert g["noise"] == pytest.approx(0.5) and g[["a", "b"]].sum() == pytest.approx(0.5)
    # a constant indicator gets nothing; the result feeds desirability_index
    ind["flat"] = 3.0
    wf = ec.pca_weights(ind, variance=0.6)
    assert wf["flat"] == 0 and wf.sum() == pytest.approx(1)
    score = ec.desirability_index(ind, wf)
    assert score.notna().all() and abs(np.corrcoef(score, common)[0, 1]) > 0.95
    with pytest.raises(ValueError):
        ec.pca_weights(ind, groups={"x": ["a", "zzz"]})
    with pytest.raises(ValueError):
        ec.pca_weights(ind, variance=0)


# ── net Complexity Outlook Gain ───────────────────────────────────────────────
def test_cog_net_is_the_exact_change_in_coi(big):
    m = ec.mcp(big)
    _, pci = ec.eci_pci(big)
    phi = ec.product_proximity(big)
    kw = {"phi": phi, "use_rca": False, "threshold": 0.5}
    net = ec.cog(m, pci, net=True, **kw)
    gross = ec.cog(m, pci, **kw)
    coi0 = ec.coi(m, pci, **kw)
    checked = 0
    for r in m.index[:6]:
        for c in m.columns[m.loc[r] == 0][:5]:
            m2 = m.copy()
            m2.loc[r, c] = 1
            assert net.loc[r, c] == pytest.approx(ec.coi(m2, pci, **kw)[r] - coi0[r], abs=1e-10)
            checked += 1
    assert checked > 10
    dens = ec.relatedness_density(m, percent=False, **kw)
    np.testing.assert_allclose(net, gross - (1 - m) * dens * pci)
    assert (net.values[m.values == 1] == 0).all()
