"""
Usage example for the econcomplex library
==========================================

A guided tour of the main indicator groups. Run from the repository root:

    python examples/basic_usage.py

Requires the library to be installed (`pip install .` or `pip install -e .`).
"""

import numpy as np
import pandas as pd
import econcomplex as ec

# ─────────────────────────────────────────────────────────────
# 1. Sample data: regions × sectors (employment)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
regions = ["North", "Northeast", "Southeast", "South", "Midwest"]
sectors = ["Agro", "Industry", "Services", "Technology", "Energy", "Health"]

# Employment matrix (R × C)
employment = pd.DataFrame(
    np.random.poisson(lam=[[50,10,200,5,20,30],
                            [80,15,150,3,10,25],
                            [20,200,500,100,50,200],
                            [40,80,180,30,25,60],
                            [30,20,100,10,40,15]]),
    index=regions,
    columns=sectors,
)
print("Employment matrix:")
print(employment)

# For larger synthetic datasets, the library ships a generator:
#   df = ec.make_sample_data(n_locs=50, n_acts=30, seed=42)
#   mat = ec.pivot_to_matrix(df, "loc", "act", "val")

# ─────────────────────────────────────────────────────────────
# 2. Long format (for the pipeline)
# ─────────────────────────────────────────────────────────────
df_long = ec.melt_matrix(employment, "region", "sector", "employment")
print(f"\nLong format: {df_long.shape} rows × columns: {list(df_long.columns)}")

# ─────────────────────────────────────────────────────────────
# 3. Pre-processing (sparse / degenerate data)
# ─────────────────────────────────────────────────────────────
# Units with zero diversity/ubiquity break or distort complexity
# calculations. ec.eci_pci() trims them automatically (trim=True) and
# returns them as NaN. The trimming is also available standalone:
core = ec.trim_core(employment)              # removes degenerate units
core22 = ec.trim_core(employment, dmin=2, umin=2)  # well-connected core
print(f"\ntrim_core: {employment.shape} -> {core.shape} "
      f"(2,2)-core: {core22.shape}")

# ─────────────────────────────────────────────────────────────
# 4. Revealed Comparative Advantage (RCA) indicators
# ─────────────────────────────────────────────────────────────
rca_mat = ec.rca(employment)
print("\nRCA (Revealed Comparative Advantage):")
print(rca_mat.round(2))

mcp_mat = ec.mcp(employment)
print("\nPresence (Mcp, RCA >= 1):")
print(mcp_mat.astype(int))

# ─────────────────────────────────────────────────────────────
# 5. Diversity and Ubiquity
# ─────────────────────────────────────────────────────────────
print("\nDiversity (no. of sectors with RCA>=1 per region):")
print(ec.diversity(employment))

print("\nUbiquity (no. of regions with RCA>=1 per sector):")
print(ec.ubiquity(employment))

# ─────────────────────────────────────────────────────────────
# 6. Economic Complexity — one function, three methods
# ─────────────────────────────────────────────────────────────
# ec.eci_pci() is the single entry point; pick the method with `method=`
# (same interface as complexity_measures() in the R economiccomplexity
# package). Default iterations for the iterative methods is 20.
eci, pci = ec.eci_pci(employment, method="eigenvector")   # default
print("\nECI — Economic Complexity Index (regions):")
print(eci.sort_values(ascending=False).round(3))

print("\nPCI — Product Complexity Index (sectors):")
print(pci.sort_values(ascending=False).round(3))

eci_mor, pci_mor = ec.eci_pci(employment, method="reflections")
print("\nECI (Method of Reflections):")
print(eci_mor.sort_values(ascending=False).round(3))

fitness, complexity = ec.eci_pci(employment, method="fitness")
print("\nRegion fitness (Fitness-Complexity):")
print(fitness.sort_values(ascending=False).round(4))

# Log scale for fitness analysis (Cristelli et al. 2015):
#   fit_log, comp_log = ec.fitness_complexity(employment, log_fitness=True)

# ─────────────────────────────────────────────────────────────
# 7. Product Space — Proximity and Density
# ─────────────────────────────────────────────────────────────
phi = ec.proximity(employment)["product"]
print("\nProximity between sectors (phi):")
print(phi.round(3))

density = ec.density(employment, phi=phi)     # alias of relatedness_density
print("\nRelatedness density (%):")
print(density.round(1))

distance = ec.distance(employment, phi=phi)
print("\nDistance (1 - density/100):")
print(distance.round(3))

dens_ext = ec.relatedness_density_external(employment, phi=phi)
print("\nExternal density (sectors not yet present):")
print(dens_ext.round(1))

# Continuous variants:
phi_cont = ec.proximity(employment, continuous=True)["product"]
phi_cos = ec.cosine_proximity(employment)
print("\nContinuous (correlation) proximity:")
print(phi_cont.round(3))

# ─────────────────────────────────────────────────────────────
# 8. Complexity Outlook (COI and COG)
# ─────────────────────────────────────────────────────────────
coi = ec.coi(employment, pci, phi=phi)        # alias of complexity_outlook_index
print("\nCOI — Complexity Outlook Index by region:")
print(coi.sort_values(ascending=False).round(4))

cog = ec.cog(employment, pci, phi=phi)        # alias of complexity_outlook_gain
print("\nCOG — Complexity Outlook Gain (sectors not present):")
print(cog.round(4))

# ─────────────────────────────────────────────────────────────
# 9. Regional Specialization
# ─────────────────────────────────────────────────────────────
print("\nAverage Location Quotient (specialization coefficient):")
print(ec.location_quotient_avg(employment).round(3))

print("\nHachman Index (similarity to national structure):")
print(ec.hachman_index(employment).round(3))

print("\nKrugman Index:")
print(ec.krugman_index(employment).round(3))

print("\nHoover Specialization Coefficient:")
print(ec.spec_coefficient(employment).round(3))

print("\nPortfolio similarity between regions (Bahar et al.):")
print(ec.export_similarity(employment).round(3))

# ─────────────────────────────────────────────────────────────
# 10. Inequality and Concentration
# ─────────────────────────────────────────────────────────────
print("\nGini index by sector (geographic concentration):")
print(ec.gini(employment).round(3))

print("\nLocational Gini by sector (Krugman):")
print(ec.locational_gini(employment).round(3))

print("\nHoover index by sector:")
print(ec.hoover_index(employment).round(2))

print("\nHerfindahl-Hirschman by region:")
print(ec.hhi(employment).round(3))             # alias of herfindahl

print("\nShannon entropy by region (diversification):")
print(ec.shannon_entropy(employment).round(3))

# ─────────────────────────────────────────────────────────────
# 11. Productivity (PRODY / EXPY / PGI)
# ─────────────────────────────────────────────────────────────
gdp_per_capita = pd.Series(
    [18000, 12000, 45000, 38000, 22000], index=regions, name="gdp"
)
print("\nPRODY — productivity level by sector:")
print(ec.prody(employment, gdp_per_capita).round(0))

print("\nEXPY — regional portfolio productivity:")
print(ec.expy(employment, gdp_per_capita).round(0))

gini_regional = pd.Series(
    [0.52, 0.55, 0.46, 0.43, 0.50], index=regions, name="gini"
)
print("\nPGI — Product Gini Index (inequality embedded in a sector):")
print(ec.pgi(employment, gini_regional).round(3))

# ─────────────────────────────────────────────────────────────
# 12. Co-occurrence indicators
# ─────────────────────────────────────────────────────────────
cooc = ec.co_occurrence(employment)
print("\nSectoral co-occurrence matrix:")
print(cooc.astype(int))

phi_jaccard = ec.relatedness_index(employment, method="jaccard")
print("\nJaccard index between sectors:")
print(phi_jaccard.round(3))

# ─────────────────────────────────────────────────────────────
# 13. Cross-space (two distinct spaces)
# ─────────────────────────────────────────────────────────────
# Patent matrix (same regions, different technologies)
patents = pd.DataFrame(
    np.random.poisson(lam=[[5,2,0],[1,10,3],[8,5,15],[2,8,4],[1,1,2]]),
    index=regions,
    columns=["Tech_A", "Tech_B", "Tech_C"],
)
x_phi = ec.cross_space_proximity(employment, patents)   # alias of cross_proximity
print("\nCross-space proximity (sectors × technologies):", x_phi.shape)
print(x_phi.round(3))

# ─────────────────────────────────────────────────────────────
# 14. Temporal dynamics (growth, entry/exit)
# ─────────────────────────────────────────────────────────────
employment_t2 = employment.copy()
employment_t2.loc["North", "Technology"] = 50   # New entry
employment_t2.loc["Northeast", "Energy"] = 0    # Exit

print("\nSector entries between t1 and t2:")
print(ec.entry([employment, employment_t2]).astype(int))

print("\nSector exits between t1 and t2:")
print(ec.exit([employment, employment_t2]).astype(int))

# Long-format (panel) wrappers:
df_t1 = ec.melt_matrix(employment, "region", "sector", "employment")
df_t1["year"] = 2020
df_t2 = ec.melt_matrix(employment_t2, "region", "sector", "employment")
df_t2["year"] = 2021
panel = pd.concat([df_t1, df_t2], ignore_index=True)

growth = ec.growth_rates(panel, "region", "sector", "employment", "year")
print("\nRegional growth rates (long format):")
print(growth.round(2))

entries = ec.entry_tracking(panel, "region", "sector", "employment", "year")
print("\nEntries (long format, only events):")
print(entries[entries["entry"] == 1])

# ─────────────────────────────────────────────────────────────
# 15. Full pipeline (one call, all indicators)
# ─────────────────────────────────────────────────────────────
print("\n=== FULL PIPELINE ===")
result = ec.compute_complexity(
    df_long,
    cols={"loc": "region", "act": "sector", "val": "employment"},
    method="eigenvector",          # or 'reflections', 'fitness'
    presence_test="rca",
    threshold=1.0,
    proximity_type="discrete",     # or 'continuous'
    proximity_method="max",        # or 'sqrt', 'min'
    compute_coi_cog=True,
)
print("Generated columns:", sorted(result.columns.tolist()))
print("\nSample output:")
print(result[["region","sector","rca","mcp","eci","pci","density","distance","coi","cog"]].head(10).round(3).to_string())

print("\n=== COMPLETED SUCCESSFULLY ===")
