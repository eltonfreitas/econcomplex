"""
From indicators to a diversification strategy
=============================================

End-to-end example of the strategy, policy-list and validation tools on
synthetic data. Run from the repository root:

    python examples/diversification_strategy.py

Steps: (1) check that relatedness predicts entry, out of sample;
(2) explain why an opportunity is within reach; (3) build a desirability
index and test its weights; (4) classify each region's options;
(5) read the result through lists of activities (a mission, a chain);
(6) check how far the complexity ranking can be trusted;
(7) with scikit-learn installed, fit a prediction model on several
layers and turn its probabilities into expected jobs.
"""

import importlib.util

import numpy as np
import pandas as pd
import econcomplex as ec

rng = np.random.default_rng(7)

# ─────────────────────────────────────────────────────────────
# 0. Two periods of synthetic data (regions x activities)
# ─────────────────────────────────────────────────────────────
mat_t0 = ec.pivot_to_matrix(
    ec.make_sample_data(n_locs=60, n_acts=40, seed=7), "loc", "act", "val")

# In the second period activities grow faster where the region already
# has related activities, so that entries follow relatedness (plus noise).
phi_t0 = ec.product_proximity(mat_t0)
density_t0 = ec.relatedness_density(mat_t0, phi=phi_t0, percent=False)
size = np.outer(mat_t0.sum(axis=1), mat_t0.sum(axis=0)) / mat_t0.values.sum()
mat_t1 = (mat_t0 * rng.lognormal(0, 0.3, mat_t0.shape)
          + size * rng.lognormal(0, 1.0, mat_t0.shape) * 3 * density_t0 ** 2)

# ─────────────────────────────────────────────────────────────
# 1. Does relatedness predict entry?
# ─────────────────────────────────────────────────────────────
rca_t0, rca_t1 = ec.rca(mat_t0), ec.rca(mat_t1)
target, at_risk = ec.entry_targets(rca_t0, rca_t1)      # RCA < 0.5 -> RCA >= 1
print(f"cells at risk: {int(at_risk.values.sum())}, "
      f"entries: {int(target.sum().sum())}")

scores = {
    "density": ec.relatedness_density(mat_t0, phi=phi_t0),
    "relative relatedness": ec.relative_relatedness(mat_t0, phi=phi_t0),
    "persistence (RCA t0)": rca_t0,
    "noise": pd.DataFrame(rng.random(mat_t0.shape), mat_t0.index, mat_t0.columns),
}
val = ec.validate_entry(scores, target, k=(5,), baseline="persistence (RCA t0)",
                        n_boot=100, seed=1)
print("\nValidation (higher is better):")
print(val["value"].unstack()[["auc", "auc_within", "precision@5", "recall@5"]].round(3))
print("\nAUC of density vs. persistence (difference and 95% interval):")
print(val.loc[("density", "auc"), ["diff", "diff_lo", "diff_hi"]].round(3).to_string())

# ─────────────────────────────────────────────────────────────
# 2. Why is an activity within reach?
# ─────────────────────────────────────────────────────────────
region = mat_t0.index[10]
options = ec.relatedness_density_external(mat_t0, phi=phi_t0)   # NaN where present
best = options.loc[region].idxmax()
print(f"\nClosest missing activity for {region}: {best} "
      f"(density {options.loc[region, best]:.1f}%)")
print("It is close because the region already has:")
print(ec.density_contributions(mat_t0, region, best, phi=phi_t0).head(4).round(2).to_string())

# Density net of "diversified regions are close to everything":
specific = ec.decomposed_density(options)
print(f"\nMost region-specific opportunity for {region}: {specific.loc[region].idxmax()}")

# ─────────────────────────────────────────────────────────────
# 3. What is worth having? Desirability with explicit weights
# ─────────────────────────────────────────────────────────────
_, pci = ec.eci_pci(mat_t0)
wage = pd.Series(1000 * np.exp(0.3 * pci.fillna(0) + rng.normal(0, 0.3, len(pci))),
                 index=pci.index)                                # illustrative
indicators = pd.DataFrame({"pci": pci, "wage": wage})
weights = {"pci": 0.6, "wage": 0.4}

desirability = ec.desirability_index(indicators, weights)
robust = ec.weight_sensitivity(indicators, weights, n_draws=500, top=10, seed=1)
print("\nMost desirable activities and how stable their rank is:")
print(robust.head(5).round(2).to_string())

# ─────────────────────────────────────────────────────────────
# 4. Feasibility x desirability quadrants, by region
# ─────────────────────────────────────────────────────────────
quadrants = ec.strategy_quadrants(options, desirability)
print(f"\nOptions of {region} by quadrant:")
print(quadrants.loc[region].value_counts().to_string())
short_jumps = quadrants.columns[quadrants.loc[region] == "short jump"]
print("Short jumps:", ", ".join(short_jumps[:8]))

# ─────────────────────────────────────────────────────────────
# 5. Lists of activities: a mission and a value chain
# ─────────────────────────────────────────────────────────────
acts = list(mat_t0.columns)
lists = {"mission": acts[25:], "everything": acts}
tab = ec.list_complexity_potential(mat_t0, pci, density_t0, lists)
print("\nRegions with the highest potential in the mission's activities:")
print(tab["potential"].unstack()["mission"].nlargest(5).round(3).to_string())

parts = ec.list_density(mat_t0, {"mission": acts[25:]}, phi=phi_t0)
share = (parts["mission"] / ec.relatedness_density(mat_t0, phi=phi_t0)).loc[region, best]
print(f"\nShare of the density of {best} in {region} that rests on mission activities: {share:.0%}")

chain = {"inputs": acts[0:6], "components": acts[15:21], "assembly": acts[33:39]}
status = ec.missing_links(mat_t0, chain, density_t0, cut=0.4)
print("\nLinks of the chain, first regions:")
print(status.head(5).to_string())
print("\nRegions by number of links present:")
print((status == "present").sum(axis=1).value_counts().sort_index().to_string())

# Activities expected to shrink (say, the first ten) and where regions can go
risk = ec.transition_risk(mat_t0, acts[:10], pci=pci, phi=phi_t0)
print("\nMost locked-in regions (exposure weighted by low complexity):")
print(risk.nlargest(4, "lock_in").round(3).to_string())

# Are each region's nearby options also its valuable ones?
alignment = ec.opportunity_alignment(options, desirability)
print(f"\nRegions whose nearby options are of low value (alignment < 0): "
      f"{int((alignment < 0).sum())} of {alignment.notna().sum()}")

# ─────────────────────────────────────────────────────────────
# 6. How far can the complexity ranking be trusted?
# ─────────────────────────────────────────────────────────────
print("\nECI diagnostics:")
print(ec.eci_diagnostics(mat_t0, n_boot=50, seed=1).round(3).to_string())

eci, _ = ec.eci_pci(mat_t0)
fitness, _ = ec.eci_pci(mat_t0, method="fitness")
genepy, _ = ec.eci_pci(mat_t0, method="genepy")
ranks = pd.DataFrame({"eci": eci, "fitness": fitness, "genepy": genepy})
print("\nRank correlation between the complexity measures:")
print(ranks.corr(method="spearman").round(2).to_string())

# ─────────────────────────────────────────────────────────────
# 7. A prediction model on several layers (needs scikit-learn)
# ─────────────────────────────────────────────────────────────
if importlib.util.find_spec("sklearn") is None:
    print("\nscikit-learn not installed: skipping the prediction model "
          "(pip install 'econcomplex[ml]')")
else:
    layers = {"density": scores["density"], "implied_rca": ec.implied_rca(mat_t0)}
    X = ec.build_features(mat_t0, layers, at_risk=at_risk)
    scores["logit (layers + controls)"] = ec.predict_entry(X, target, model="logit", seed=1)
    scores["gbm (layers + controls)"] = ec.predict_entry(X, target, model="gbm", seed=1)
    val = ec.validate_entry(scores, target, k=(5,))["value"].unstack()
    print("\nOut-of-sample comparison, models included:")
    print(val[["auc", "auc_within", "precision@5"]].round(3).to_string())

    print("\nWhat the model relies on (fall in AUC when the feature is shuffled):")
    print(ec.explain_entry(X, target, model="logit", seed=1).round(4).to_string())

    # Next window: fit on t0 -> t1, predict from what is observed at t1
    phi_t1 = ec.product_proximity(mat_t1)
    layers_t1 = {"density": ec.relatedness_density(mat_t1, phi=phi_t1),
                 "implied_rca": ec.implied_rca(mat_t1)}
    X_t1 = ec.build_features(mat_t1, layers_t1, at_risk=rca_t1 < 0.5)
    prob = ec.predict_entry(X, target, model="logit", new_features=X_t1, calibrate=True)
    jobs = ec.potential_employment(prob, mat_t1)
    print("\nExpected size of the entries of the next window, top regions:")
    print(jobs.groupby("region")["expected"].sum().nlargest(5).round(1).to_string())
