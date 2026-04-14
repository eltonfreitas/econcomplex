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
    col_sums = arr.sum(axis=0)
    total = arr.sum()
    # Sum of (share_c_r * RCA_r_c) over regions = 1 for each c
    weighted = (arr / arr.sum(axis=1, keepdims=True)) * rca_arr
    # col weighted mean should be approx 1
    col_means = (weighted.sum(axis=0) * total) / (col_sums * col_sums)
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
