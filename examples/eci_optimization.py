"""
ECI Optimization and strategic diffusion — step-by-step example
================================================================

Demonstrates the optimization layer of econcomplex:

1. ECI Optimization (Stojkoski & Hidalgo 2026): calibrate a
   stepping-stone forecast model on a panel, compute the effort each
   new specialization requires, and select minimal-effort portfolios
   that reach an ECI target.
2. Growth targeting: convert a growth target into an ECI target.
3. Strategic diffusion (Alshamsi, Pinheiro & Hidalgo 2018): when to
   make unrelated bets — compare diversification strategies on the
   product space and optimize the entry sequence.

Run from the repository root:

    python examples/eci_optimization.py

The example uses synthetic data so it runs anywhere; replace the panel
with your own long-format data (location, activity, value, year).
"""

import warnings

import numpy as np
import pandas as pd
import econcomplex as ec

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# 1. Synthetic panel with three periods (t, t+5, t+10)
# ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
df0 = ec.make_sample_data(n_locs=30, n_acts=20, seed=0)
mats = {2000: ec.pivot_to_matrix(df0, "loc", "act", "val")}
for y0, y1 in [(2000, 2005), (2005, 2010)]:
    noise = rng.lognormal(mean=0.0, sigma=0.3, size=mats[y0].shape)
    mats[y1] = mats[y0] * noise * 1.1

frames = []
for year, m in mats.items():
    d = ec.melt_matrix(m, "loc", "act", "val")
    d = d[d["val"] > 0].copy()
    d["year"] = year
    frames.append(d)
panel = pd.concat(frames, ignore_index=True)
print(f"Panel: {panel.shape[0]} rows, periods {sorted(panel.year.unique())}")

# ─────────────────────────────────────────────────────────────
# 2. Calibrate the stepping-stone model (eq. 1 of the paper)
# ─────────────────────────────────────────────────────────────
model = ec.calibrate_steppingstone(
    panel, "loc", "act", "val", "year", horizon=10, steppingstone=5,
)
print("\nEntry-model coefficients:")
for k, v in model["entry"].items():
    print(f"  {k:28s} {v:8.4f}")

# ─────────────────────────────────────────────────────────────
# 3. Effort and no-policy forecast for the latest year
# ─────────────────────────────────────────────────────────────
mat = mats[2010]
W = ec.effort_matrix(mat, model)            # added RCA per candidate entry
forecast = ec.forecast_specialization(mat, model)
print("\nProjected ECI (first 5 locations):")
print(forecast["eci"].head().round(3))

# ─────────────────────────────────────────────────────────────
# 4. Minimal-effort portfolios for an ECI increase of +0.1
# ─────────────────────────────────────────────────────────────
portfolio = ec.eci_optimization(mat, model, delta_eci=0.1)
print(f"\nECI Optimization: {len(portfolio)} suggestions "
      f"for {portfolio['location'].nunique()} locations")
first = portfolio[portfolio["location"] == portfolio["location"].iloc[0]]
print(first.round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 5. Growth targeting (eq. 3): growth target -> ECI target
# ─────────────────────────────────────────────────────────────
# Synthetic macro panel: GDP per capita grows with ECI (convergence too)
locs = list(mat.index)
gdppc = pd.Series(rng.uniform(2e3, 5e4, len(locs)), index=locs)
eci_panel = pd.Series(rng.normal(0, 1, len(locs)), index=locs)
macro_rows = []
for year in (2000, 2010, 2020):
    for c in locs:
        macro_rows.append({"loc": c, "year": year,
                           "gdppc": gdppc[c], "eci": eci_panel[c]})
    z = (np.log(gdppc) - np.log(gdppc).mean()) / np.log(gdppc).std()
    g = 0.02 + 0.01 * eci_panel - 0.005 * z + rng.normal(0, 0.002, len(locs))
    gdppc = gdppc * np.exp(10 * g)
    eci_panel = eci_panel + rng.normal(0, 0.1, len(locs))
macro = pd.DataFrame(macro_rows)

gm = ec.calibrate_growth_model(macro, "loc", "year", "gdppc", "eci",
                               horizon=10)
eci_star = ec.eci_target_for_growth(gm, growth_target=0.035,
                                    gdppc_now=15000.0)
print(f"\nECI compatible with 3.5%/yr growth (GDPpc 15k): {eci_star:.3f}")
print(f"Check (expected growth at that ECI): "
      f"{ec.expected_growth(gm, eci_star, 15000.0):.4f}")

# ─────────────────────────────────────────────────────────────
# 6. Strategic diffusion: when to make the unrelated bet
# ─────────────────────────────────────────────────────────────
adj = ec.proximity_network(mat, phi_threshold=0.4)
fit = ec.calibrate_contagion(panel, "loc", "act", "val", "year",
                             adjacency=adj)
print(f"\nContagion calibration: p = {fit['B']:.3f} * x^{fit['alpha']:.2f} "
      f"({fit['n_events']} entry events)")

loc0 = mat.index[0]
active0 = ec.mcp(mat).reindex(columns=adj.index, fill_value=0.0).loc[loc0]

table = ec.compare_strategies(adj, active0, B=fit["B"], alpha=fit["alpha"])
print(f"\nDiversification strategies for {loc0} "
      "(expected total time, lower is better):")
print(table.round(1))

best = ec.optimize_sequence(adj, active0, B=fit["B"], alpha=fit["alpha"],
                            n_iter=1000, seed=0)
print(f"\nOptimized sequence: total time {best['total_time']:.1f} "
      f"vs greedy {best['greedy_time']:.1f} "
      f"(improvement {best['improvement']:.1f})")
print("First targets:", best["sequence"][:5])

print("\n=== COMPLETED SUCCESSFULLY ===")
