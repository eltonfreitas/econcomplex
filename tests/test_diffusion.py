"""Tests for strategic diffusion (Alshamsi, Pinheiro & Hidalgo 2018)."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import econcomplex as ec


# ── Wheel network (the paper's analytical benchmark) ──────────────────────────
Z = 30      # 1 hub + 29 peripherals
ALPHA = 1.5


@pytest.fixture(scope="module")
def wheel():
    """Hub (node 0) connected to all peripherals; peripherals form a ring."""
    n = Z
    adj = np.zeros((n, n))
    adj[0, 1:] = adj[1:, 0] = 1.0
    for i in range(1, n):
        j = i + 1 if i < n - 1 else 1
        adj[i, j] = adj[j, i] = 1.0
    labels = [f"N{i:02d}" for i in range(n)]
    return pd.DataFrame(adj, index=labels, columns=labels)


@pytest.fixture(scope="module")
def seed_state(wheel):
    active0 = pd.Series(0.0, index=wheel.index)
    active0["N01"] = 1.0  # a single peripheral, as in the paper
    return active0


def wheel_sequence(L):
    """Activate L-1 more peripherals contiguously, then the hub, then the rest."""
    periph = [f"N{i:02d}" for i in range(2, Z)]
    return periph[:L - 1] + ["N00"] + periph[L - 1:]


def eq2_time(L, alpha):
    """Closed-form total time, eq. 2 of Alshamsi et al. (2018)."""
    return (3 ** alpha * (L - 1)
            + ((Z - 1) / L) ** alpha
            + (3 / 2) ** alpha * (Z - 2 - L)
            + 1)


def test_expected_time_matches_paper_eq2(wheel, seed_state):
    for L in (2, 5, 10, 20):
        t_lib = ec.expected_diversification_time(
            wheel, seed_state, wheel_sequence(L), B=1.0, alpha=ALPHA)
        assert np.isclose(t_lib, eq2_time(L, ALPHA)), f"L={L}"


def test_greedy_targets_hub_too_late(wheel, seed_state):
    """Suboptimality of relatedness: the optimal hub window comes earlier
    than the greedy strategy's, and yields a lower total time."""
    times = {L: eq2_time(L, ALPHA) for L in range(1, Z - 1)}
    L_star = min(times, key=times.get)

    greedy = ec.diversification_strategy(wheel, seed_state, B=1.0,
                                         alpha=ALPHA, strategy="greedy")
    L_greedy = int(np.flatnonzero(greedy["activity"].values == "N00")[0]) + 1
    t_greedy = float(greedy["cumulative_time"].iloc[-1])

    assert L_star < L_greedy
    assert times[L_star] < t_greedy


def test_hub_too_early_performs_poorly(wheel, seed_state):
    res = ec.compare_strategies(wheel, seed_state, B=1.0, alpha=ALPHA)
    assert (res["n_activated"] == Z - 1).all()
    assert res.loc["greedy", "total_time"] < res.loc["high_degree", "total_time"]


def test_optimize_sequence_beats_greedy(wheel, seed_state):
    res = ec.optimize_sequence(wheel, seed_state, B=1.0, alpha=ALPHA,
                               n_iter=3000, seed=0)
    assert res["total_time"] <= res["greedy_time"] + 1e-9
    assert res["improvement"] > 0  # the wheel greedy is strictly suboptimal
    # the optimized sequence is a valid permutation of the reachable nodes
    assert sorted(res["sequence"]) == sorted(wheel.index.drop("N01"))
    # and the hub is targeted earlier than in the greedy sequence
    greedy = ec.diversification_strategy(wheel, seed_state, B=1.0,
                                         alpha=ALPHA, strategy="greedy")
    pos_opt = res["sequence"].index("N00")
    pos_greedy = int(np.flatnonzero(greedy["activity"].values == "N00")[0])
    assert pos_opt < pos_greedy


def test_activation_probabilities_wheel(wheel, seed_state):
    p = ec.activation_probabilities(wheel, seed_state, B=1.0, alpha=ALPHA)
    assert p["N01"] == 0.0                               # already active
    assert np.isclose(p["N02"], (1 / 3) ** ALPHA)        # ring neighbor
    assert np.isclose(p["N00"], (1 / (Z - 1)) ** ALPHA)  # hub
    assert p["N05"] == 0.0                               # no active neighbor


# ── Proximity network ─────────────────────────────────────────────────────────
def test_proximity_network_properties():
    df = ec.make_sample_data(n_locs=30, n_acts=15, seed=2)
    mat = ec.pivot_to_matrix(df, "loc", "act", "val")
    adj = ec.proximity_network(mat, phi_threshold=0.3)
    assert set(np.unique(adj.values)).issubset({0.0, 1.0})
    assert np.allclose(adj.values, adj.values.T)
    assert np.allclose(np.diag(adj.values), 0.0)


# ── Contagion calibration ─────────────────────────────────────────────────────
def test_calibrate_contagion_recovers_parameters():
    """Generate entries with p = 0.3 * x^1.5 and recover B and alpha."""
    rng = np.random.default_rng(7)
    n_acts, n_locs = 40, 80
    acts = [f"A{i:02d}" for i in range(n_acts)]
    locs = [f"L{i:02d}" for i in range(n_locs)]

    adj = (rng.random((n_acts, n_acts)) < 0.15).astype(float)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    adjacency = pd.DataFrame(adj, index=acts, columns=acts)
    k = adj.sum(axis=1)

    M = (rng.random((n_locs, n_acts)) < 0.25).astype(float)
    frames = []
    for year in range(4):
        d = ec.melt_matrix(pd.DataFrame(M, index=locs, columns=acts),
                           "loc", "act", "val")
        d = d[d["val"] > 0].copy()
        d["year"] = year
        frames.append(d)
        x = np.divide(M @ adj, k[None, :], out=np.zeros_like(M),
                      where=k[None, :] > 0)
        p = 0.3 * x ** 1.5
        M = np.maximum(M, (rng.random(M.shape) < p).astype(float))
    panel = pd.concat(frames, ignore_index=True)

    fit = ec.calibrate_contagion(panel, "loc", "act", "val", "year",
                                 adjacency=adjacency, presence_test="manual")
    assert fit["n_events"] > 50
    assert 1.1 < fit["alpha"] < 1.9
    assert 0.15 < fit["B"] < 0.6
    assert len(fit["bins"]) > 3
