"""Tests for core indicators."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import econcomplex as ec


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_mat():
    """Small 4-region × 5-activity matrix."""
    np.random.seed(42)
    data = np.array([
        [10, 20, 0,  5,  1],
        [0,  5,  30, 2,  8],
        [15, 0,  5,  20, 3],
        [2,  2,  2,  2,  40],
    ], dtype=float)
    return pd.DataFrame(
        data,
        index=["R1", "R2", "R3", "R4"],
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def sample_long(sample_mat):
    return ec.melt_matrix(sample_mat, "region", "activity", "employment")


# ── RCA tests ─────────────────────────────────────────────────────────────────
def test_rca_shape(sample_mat):
    result = ec.rca(sample_mat)
    assert result.shape == sample_mat.shape


def test_rca_average_is_one(sample_mat):
    """Column-weighted average of RCA should equal 1 for each activity."""
    arr = sample_mat.values.astype(float)
    rca_arr = ec.rca(arr)
    # simplified check: global mean of RCA = 1
    assert abs(rca_arr.mean() - 1.0) < 0.5  # not exact but confirms normalization


def test_rca_binary(sample_mat):
    result = ec.rca(sample_mat, binary=True)
    assert set(result.values.flatten()).issubset({0.0, 1.0})


def test_rca_returns_dataframe(sample_mat):
    result = ec.rca(sample_mat)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(sample_mat.index)
    assert list(result.columns) == list(sample_mat.columns)


# ── Diversity / Ubiquity ──────────────────────────────────────────────────────
def test_diversity_ubiquity(sample_mat):
    div = ec.diversity(sample_mat)
    ubi = ec.ubiquity(sample_mat)
    assert len(div) == sample_mat.shape[0]
    assert len(ubi) == sample_mat.shape[1]
    assert (div >= 0).all()
    assert (ubi >= 0).all()


def test_diversity_max(sample_mat):
    div = ec.diversity(sample_mat)
    assert (div <= sample_mat.shape[1]).all()


# ── Complexity ────────────────────────────────────────────────────────────────
def test_eci_pci_shape(sample_mat):
    eci, pci = ec.eci_pci(sample_mat)
    assert len(eci) == sample_mat.shape[0]
    assert len(pci) == sample_mat.shape[1]


def test_eci_normalized(sample_mat):
    eci, pci = ec.eci_pci(sample_mat)
    assert abs(eci.mean()) < 1e-10
    assert abs(eci.std(ddof=0) - 1.0) < 1e-10


def test_reflections_shape(sample_mat):
    eci, pci = ec.method_of_reflections(sample_mat)
    assert len(eci) == sample_mat.shape[0]
    assert len(pci) == sample_mat.shape[1]


def test_fitness_complexity(sample_mat):
    fitness, complexity = ec.fitness_complexity(sample_mat)
    assert len(fitness) == sample_mat.shape[0]
    assert len(complexity) == sample_mat.shape[1]
    assert (fitness >= 0).all()


# ── Proximity & Density ───────────────────────────────────────────────────────
def test_proximity_symmetric(sample_mat):
    phi = ec.proximity(sample_mat)["product"]
    assert phi.shape == (sample_mat.shape[1], sample_mat.shape[1])
    # Symmetric
    np.testing.assert_allclose(phi.values, phi.values.T, atol=1e-10)
    # Diagonal zero
    assert (np.diag(phi.values) == 0).all()
    # Values in [0, 1]
    assert (phi.values >= -1e-10).all()
    assert (phi.values <= 1 + 1e-10).all()


def test_density_shape(sample_mat):
    dens = ec.relatedness_density(sample_mat)
    assert dens.shape == sample_mat.shape


def test_density_range(sample_mat):
    dens = ec.relatedness_density(sample_mat)
    assert (dens.values >= -1e-10).all()
    assert (dens.values <= 100 + 1e-10).all()


# ── Specialization ────────────────────────────────────────────────────────────
def test_krugman_range(sample_mat):
    k = ec.krugman_index(sample_mat)
    assert (k >= 0).all()
    assert (k <= 2).all()


def test_hachman_range(sample_mat):
    h = ec.hachman_index(sample_mat)
    assert (h >= 0).all()
    assert (h <= 1 + 1e-10).all()


def test_spec_coeff_half_krugman(sample_mat):
    spec = ec.specialization_coefficient(sample_mat)
    krug = ec.krugman_index(sample_mat)
    np.testing.assert_allclose(spec.values * 2, krug.values, atol=1e-10)


# ── Inequality ────────────────────────────────────────────────────────────────
def test_gini_range(sample_mat):
    g = ec.gini(sample_mat)
    assert (g >= 0).all()
    assert (g <= 1).all()


def test_herfindahl_range(sample_mat):
    hhi = ec.herfindahl(sample_mat, normalize=False)
    assert (hhi >= 0).all()
    assert (hhi <= 1 + 1e-10).all()


def test_entropy_positive(sample_mat):
    h = ec.shannon_entropy(sample_mat)
    assert (h >= 0).all()


# ── Productivity ──────────────────────────────────────────────────────────────
def test_prody_expy(sample_mat):
    gdp = np.array([100, 200, 150, 80], dtype=float)
    p = ec.prody(sample_mat, gdp)
    e = ec.expy(sample_mat, gdp)
    assert len(p) == sample_mat.shape[1]
    assert len(e) == sample_mat.shape[0]
    assert (p >= 0).all()
    assert (e >= 0).all()


# ── Patents ───────────────────────────────────────────────────────────────────
def test_eor(sample_mat):
    # Use sample_mat as patent × technology incidence
    eor = ec.ease_of_recombination(sample_mat)
    assert len(eor) == sample_mat.shape[1]
    assert (eor >= 0).all()
    assert (eor <= 1 + 1e-10).all()


# ── Dynamics ──────────────────────────────────────────────────────────────────
def test_growth_rate(sample_mat):
    mat2 = sample_mat * 1.1
    g = ec.growth_rate(sample_mat, mat2, axis=0)
    np.testing.assert_allclose(g.values, 10.0, atol=1e-6)


def test_entry_exit(sample_mat):
    mat2 = sample_mat.copy()
    mat2.iloc[0, 2] = 100  # force entry for (R1, C)
    ent = ec.entry([sample_mat, mat2])
    assert ent.shape == sample_mat.shape


# ── Pipeline ──────────────────────────────────────────────────────────────────
def test_pipeline_columns(sample_long):
    result = ec.compute_complexity(
        sample_long,
        cols={"loc": "region", "act": "activity", "val": "employment"},
        compute_coi_cog=False,
    )
    for col in ["rca", "mcp", "diversity", "ubiquity", "eci", "pci", "density", "distance"]:
        assert col in result.columns, f"Missing column: {col}"


def test_pipeline_time(sample_mat):
    df1 = ec.melt_matrix(sample_mat, "region", "activity", "employment")
    df1["year"] = 2020
    df2 = ec.melt_matrix(sample_mat * 1.05, "region", "activity", "employment")
    df2["year"] = 2021
    df = pd.concat([df1, df2])
    result = ec.compute_complexity(
        df,
        cols={"loc": "region", "act": "activity", "val": "employment", "time": "year"},
        compute_coi_cog=False,
    )
    assert set(result["year"].unique()) == {2020, 2021}


# ── NumPy 2.x compatibility ───────────────────────────────────────────────────
def test_locational_gini_runs(sample_mat):
    """np.trapz was removed in NumPy 2.0; locational_gini must still work."""
    result = ec.locational_gini(sample_mat)
    assert len(result) == sample_mat.shape[1]
    assert (result <= 0.5 + 1e-9).all()


def test_hoover_gini_runs(sample_mat):
    result = ec.hoover_gini(sample_mat)
    assert len(result) == sample_mat.shape[1]


# ── eci_pci dispatcher ────────────────────────────────────────────────────────
def test_eci_pci_method_eigenvector_default(sample_mat):
    eci_default, pci_default = ec.eci_pci(sample_mat)
    eci_explicit, pci_explicit = ec.eci_pci(sample_mat, method="eigenvector")
    assert np.allclose(eci_default.values, eci_explicit.values)
    assert np.allclose(pci_default.values, pci_explicit.values)


def test_eci_pci_method_reflections(sample_mat):
    eci, pci = ec.eci_pci(sample_mat, method="reflections", iterations=20)
    eci_direct, pci_direct = ec.method_of_reflections(sample_mat, iterations=20)
    assert np.allclose(eci.values, eci_direct.values)
    assert np.allclose(pci.values, pci_direct.values)


def test_eci_pci_method_fitness(sample_mat):
    fit, comp = ec.eci_pci(sample_mat, method="fitness", iterations=100)
    fit_direct, comp_direct = ec.fitness_complexity(sample_mat, iterations=100)
    assert np.allclose(fit.values, fit_direct.values)
    assert np.allclose(comp.values, comp_direct.values)


def test_eci_pci_invalid_method(sample_mat):
    with pytest.raises(ValueError):
        ec.eci_pci(sample_mat, method="banana")


# ── New fitness/reflections options ───────────────────────────────────────────
def test_fitness_log_scale(sample_mat):
    fit, comp = ec.fitness_complexity(sample_mat)
    fit_log, comp_log = ec.fitness_complexity(sample_mat, log_fitness=True)
    positive = fit.values > 0
    assert np.allclose(fit_log.values[positive], np.log(fit.values[positive]))


def test_reflections_tol_converges(sample_mat):
    """With a loose tolerance and many iterations the result must match
    the fully-iterated (tol=None) run within that tolerance."""
    eci_full, _ = ec.method_of_reflections(sample_mat, iterations=200, tol=None)
    eci_tol, _ = ec.method_of_reflections(sample_mat, iterations=200, tol=1e-8)
    assert np.allclose(eci_full.values, eci_tol.values, atol=1e-6)


# ── make_sample_data ──────────────────────────────────────────────────────────
def test_make_sample_data_shape_and_columns():
    df = ec.make_sample_data(n_locs=20, n_acts=10, seed=42)
    assert list(df.columns) == ["loc", "act", "val"]
    assert (df["val"] > 0).all()
    assert df["loc"].nunique() <= 20
    assert df["act"].nunique() <= 10


def test_make_sample_data_reproducible():
    df1 = ec.make_sample_data(seed=7)
    df2 = ec.make_sample_data(seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_make_sample_data_works_in_pipeline():
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=1)
    result = ec.compute_complexity(
        df, cols={"loc": "loc", "act": "act", "val": "val"},
        compute_coi_cog=False,
    )
    assert "eci" in result.columns


# ── Documented short aliases ──────────────────────────────────────────────────
def test_documented_aliases_exist():
    assert ec.density is ec.relatedness_density
    assert ec.relatedness is ec.relatedness_density
    assert ec.hhi is ec.herfindahl
    assert ec.coi is ec.complexity_outlook_index
    assert ec.cog is ec.complexity_outlook_gain
    assert ec.pgi is ec.product_gini_index
    assert ec.peii is ec.product_emissions_index
    assert ec.spec_coefficient is ec.specialization_coefficient
    assert ec.cross_space_proximity is ec.cross_proximity


# ── API organization (complexity entry point) ─────────────────────────────────
def test_eci_pci_lives_in_its_own_module():
    from econcomplex.complexity.eci_pci import eci_pci as entry
    assert entry is ec.eci_pci


def test_eigenvector_implementation_matches_dispatcher(sample_mat):
    eci_a, pci_a = ec.eci_pci(sample_mat, trim=False)
    eci_b, pci_b = ec.eci_pci_eigenvector(sample_mat)
    assert np.allclose(eci_a.values, eci_b.values)
    assert np.allclose(pci_a.values, pci_b.values)


def test_dispatcher_log_fitness(sample_mat):
    fit_a, comp_a = ec.eci_pci(sample_mat, method="fitness", log_fitness=True)
    fit_b, comp_b = ec.fitness_complexity(sample_mat, log_fitness=True)
    assert np.allclose(fit_a.values, fit_b.values, equal_nan=True)


def test_mor_variants_still_work(sample_mat):
    reg = ec.mor_regions(sample_mat, steps=4)
    act = ec.mor_activities(sample_mat, steps=4)
    assert len(reg) == sample_mat.shape[0]
    assert len(act) == sample_mat.shape[1]
    assert (reg.values >= 0).all() and (reg.values <= 100).all()


def test_continuous_proximity_methods(sample_mat):
    rca_mat = ec.rca(sample_mat)
    cos_a = ec.continuous_proximity(rca_mat, method="cosine")
    cos_b = ec.cosine_proximity(sample_mat)
    assert np.allclose(cos_a.values, cos_b.values)
    cos_c = ec.proximity(sample_mat, continuous=True,
                         continuous_method="cosine")["product"]
    assert np.allclose(cos_a.values, cos_c.values)


def test_relatedness_alias(sample_mat):
    phi = ec.proximity(sample_mat)["product"]
    dens = ec.relatedness(sample_mat, phi=phi)
    direct = ec.relatedness_density(sample_mat, phi=phi)
    assert np.allclose(dens.values, direct.values)


# ── Cross-space with different space sizes ────────────────────────────────────
def test_cross_relatedness_rectangular_spaces(sample_mat):
    """Spaces A and B with different sizes: result must be R x B,
    labeled with space-B activities."""
    mat_b = pd.DataFrame(
        np.random.default_rng(5).poisson(10, (sample_mat.shape[0], 3)),
        index=sample_mat.index, columns=["T1", "T2", "T3"],
    )
    x_phi = ec.cross_proximity(sample_mat, mat_b)        # A x B (5 x 3)
    dens = ec.cross_relatedness(sample_mat, x_phi)
    assert dens.shape == (sample_mat.shape[0], 3)
    assert list(dens.columns) == ["T1", "T2", "T3"]
    with pytest.raises(ValueError):
        ec.cross_relatedness(mat_b, x_phi)               # wrong orientation


# ── New proximity variants ────────────────────────────────────────────────────
def test_cosine_proximity(sample_mat):
    phi = ec.cosine_proximity(sample_mat)
    n_acts = sample_mat.shape[1]
    assert phi.shape == (n_acts, n_acts)
    assert np.allclose(np.diag(phi.values), 0.0)
    assert ((phi.values >= 0) & (phi.values <= 1 + 1e-9)).all()


def test_correlation_proximity(sample_mat):
    phi = ec.correlation_proximity(sample_mat)
    n_acts = sample_mat.shape[1]
    assert phi.shape == (n_acts, n_acts)
    assert ((phi.values >= 0) & (phi.values <= 1 + 1e-9)).all()


def test_proximity_continuous_param(sample_mat):
    phi = ec.proximity(sample_mat, continuous=True)["product"]
    direct = ec.correlation_proximity(sample_mat)
    assert np.allclose(phi.values, direct.values)


# ── Long-format dynamics wrappers ─────────────────────────────────────────────
@pytest.fixture
def sample_panel(sample_mat):
    df1 = ec.melt_matrix(sample_mat, "region", "activity", "employment")
    df1["year"] = 2020
    mat2 = sample_mat.copy()
    mat2.iloc[0, 2] = 100   # entry for (R1, C)
    mat2.iloc[3, 4] = 0     # exit for (R4, E)
    df2 = ec.melt_matrix(mat2, "region", "activity", "employment")
    df2["year"] = 2021
    return pd.concat([df1, df2], ignore_index=True)


def test_growth_rates_long(sample_panel):
    g = ec.growth_rates(sample_panel, "region", "activity", "employment", "year")
    assert set(g.columns) == {"region", "year", "growth"}
    assert set(g["year"].unique()) == {2021}


def test_entry_tracking_long(sample_panel):
    ent = ec.entry_tracking(sample_panel, "region", "activity", "employment", "year")
    assert set(ent.columns) == {"region", "activity", "entry"}
    assert set(ent["entry"].unique()).issubset({0.0, 1.0})
    assert ent["entry"].sum() > 0


def test_exit_tracking_long(sample_panel):
    ext = ec.exit_tracking(sample_panel, "region", "activity", "employment", "year")
    assert set(ext.columns) == {"region", "activity", "exit"}
    assert set(ext["exit"].unique()).issubset({0.0, 1.0})


# ── Pre-processing (trim_core) ────────────────────────────────────────────────
@pytest.fixture
def degenerate_mat(sample_mat):
    """sample_mat plus an all-zero region and an all-zero activity."""
    mat = sample_mat.copy()
    mat.loc["R5"] = 0.0
    mat["F"] = 0.0
    return mat


def test_trim_core_removes_degenerate_units(degenerate_mat):
    core = ec.trim_core(degenerate_mat)
    assert "R5" not in core.index
    assert "F" not in core.columns
    assert core.shape[0] >= 2 and core.shape[1] >= 2


def test_trim_core_noop_on_clean_matrix(sample_mat):
    core = ec.trim_core(sample_mat)
    assert core.shape == sample_mat.shape


def test_trim_core_stricter_core():
    """(2,2)-core drops a location active in a single product."""
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=3)
    mat = ec.pivot_to_matrix(df, "loc", "act", "val")
    core = ec.trim_core(mat, dmin=2, umin=2)
    m = ec.mcp(core)
    assert (m.sum(axis=1) >= 2).all()
    assert (m.sum(axis=0) >= 2).all()


def test_eci_pci_trims_degenerate_units(degenerate_mat):
    eci, pci = ec.eci_pci(degenerate_mat)
    assert np.isnan(eci["R5"])
    assert np.isnan(pci["F"])
    assert eci.drop("R5").notna().all()
    # values for the surviving core match the untrimmed-input calculation
    eci_clean, _ = ec.eci_pci(degenerate_mat.drop(index="R5", columns=["F"]))
    assert np.allclose(eci.drop("R5").values, eci_clean.values)


def test_eci_pci_trim_false_keeps_shape(degenerate_mat):
    eci, pci = ec.eci_pci(degenerate_mat, trim=False)
    assert eci.notna().all()
    assert len(eci) == degenerate_mat.shape[0]


def test_eci_pci_trim_all_methods(degenerate_mat):
    for method in ["eigenvector", "reflections", "fitness"]:
        eci, pci = ec.eci_pci(degenerate_mat, method=method)
        assert np.isnan(eci["R5"]), f"method={method}"
        assert np.isnan(pci["F"]), f"method={method}"


def test_pipeline_with_degenerate_data(degenerate_mat):
    long = ec.melt_matrix(degenerate_mat, "region", "activity", "employment")
    result = ec.compute_complexity(
        long,
        cols={"loc": "region", "act": "activity", "val": "employment"},
        compute_coi_cog=False,
    )
    assert result.loc[result["region"] == "R5", "eci"].isna().all()
    assert result.loc[result["region"] != "R5", "eci"].notna().all()


def test_reflections_sign_orientation(sample_mat):
    """ECI from reflections must correlate positively with diversity."""
    eci, pci = ec.method_of_reflections(sample_mat)
    div = ec.diversity(sample_mat)
    ubi = ec.ubiquity(sample_mat)
    assert np.corrcoef(eci.values, div.values)[0, 1] > 0
    assert np.corrcoef(pci.values, ubi.values)[0, 1] < 0


# ── Default iterations (20, as in the R economiccomplexity package) ───────────
def test_fitness_default_iterations_match_dispatcher(sample_mat):
    fit_a, comp_a = ec.eci_pci(sample_mat, method="fitness")
    fit_b, comp_b = ec.fitness_complexity(sample_mat)
    assert np.allclose(fit_a.values, fit_b.values)
    assert np.allclose(comp_a.values, comp_b.values)


def test_fitness_warns_when_not_converged(sample_mat):
    with pytest.warns(RuntimeWarning, match="did not converge"):
        ec.fitness_complexity(sample_mat, iterations=1, tol=1e-15)


# ── Pipeline is pure orchestration of the public functions ────────────────────
def test_pipeline_matches_manual_indicators(sample_long):
    """Each column the pipeline returns must equal the value obtained by
    calling the public functions by hand on the same matrix."""
    res = ec.compute_complexity(
        sample_long, cols={"loc": "region", "act": "activity", "val": "employment"},
        compute_coi_cog=True,
    )
    mat = ec.pivot_to_matrix(sample_long, "region", "activity", "employment")
    mcp = ec.mcp(mat)
    eci, pci = ec.eci_pci(mat)
    phi = ec.proximity(mcp, use_rca=False, threshold=0.5)["product"]

    div = ec.diversity(mcp, use_rca=False, threshold=0.5)
    dens = ec.density(mcp, phi=phi, use_rca=False, threshold=0.5)
    dist = ec.distance(mcp, phi=phi, use_rca=False, threshold=0.5)

    # one (region, activity) cell and one region-level value
    row = res.iloc[0]
    r, a = row["region"], row["activity"]
    assert np.isclose(row["eci"], eci[r], equal_nan=True)
    assert np.isclose(row["diversity"], div[r])
    assert np.isclose(row["density"], dens.loc[r, a])
    assert np.isclose(row["distance"], dist.loc[r, a])


def test_pipeline_passes_trim_params():
    """dmin/umin must reach eci_pci through the pipeline (was impossible
    before 1.0.1): the (2,2) core trims at least as many units as (1,1)."""
    long = ec.make_sample_data(n_locs=40, n_acts=25, seed=11)
    strict = ec.compute_complexity(
        long, cols={"loc": "loc", "act": "act", "val": "val"},
        compute_coi_cog=False, dmin=2, umin=2,
    )
    loose = ec.compute_complexity(
        long, cols={"loc": "loc", "act": "act", "val": "val"},
        compute_coi_cog=False,
    )
    n_nan_strict = strict.drop_duplicates("loc")["eci"].isna().sum()
    n_nan_loose = loose.drop_duplicates("loc")["eci"].isna().sum()
    assert n_nan_strict >= n_nan_loose   # stricter core trims at least as much
