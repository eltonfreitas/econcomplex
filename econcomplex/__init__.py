"""
econcomplex — Economic Complexity and Geographic Indicators Library
===================================================================

A consolidated Python library for computing Economic Complexity indicators
and Geographic/Regional Science metrics, combining the best implementations
from EconGeo (R), economiccomplexity (R), py-ecomplexity, and
py-economic-complexity.

Quick start
-----------
>>> import econcomplex as ec
>>> import pandas as pd
>>>
>>> # Long-format data: location × activity × value
>>> df = pd.read_csv("my_data.csv")
>>>
>>> # Full pipeline (single period)
>>> result = ec.compute_complexity(
...     df,
...     cols={"loc": "region", "act": "sector", "val": "employment"},
...     method="eigenvector",
... )
>>>
>>> # Individual indicators
>>> mat = ec.pivot_to_matrix(df, "region", "sector", "employment")
>>> rca_mat    = ec.rca(mat)
>>> eci, pci   = ec.eci_pci(mat)
>>> phi        = ec.proximity(mat)["product"]
>>> density    = ec.relatedness_density(mat, phi=phi)

Submodules
----------
core          : RCA, RPOP, Mcp, diversity, ubiquity, trim_core, utilities
complexity    : eci_pci() — single entry point (eigenvector, reflections,
                fitness) — plus the method-specific implementations and
                subnational ECI
relatedness   : proximity (discrete/continuous/cosine/correlation),
                relatedness density, distance, co-occurrence,
                z-score novelty, cross-space proximity/relatedness
specialization: location quotient, Hachman, Krugman, specialization coeff,
                export similarity
inequality    : Gini, locational Gini, Hoover-Gini, Hoover index,
                Herfindahl-Hirschman, Shannon entropy
productivity  : PRODY, EXPY, Product Gini Index, PEII
patents       : ease of recombination, modular complexity
dynamics      : growth rates, entry/exit tracking (matrix and panel APIs)
outlook       : COI, COG (Complexity Outlook)
optimization  : ECI Optimization (Stojkoski & Hidalgo 2026), growth
                targeting, strategic diffusion (Alshamsi et al. 2018)
pipeline      : high-level compute_complexity() function

Note: short names like density, hhi, coi, cog, pgi are aliases bound to
the same objects as their canonical functions (see the API map in the
documentation).
"""

__version__ = "1.0.1"
__author__ = "Elton Freitas and contributors"

# ── Core ──────────────────────────────────────────────────────────────────────
from .core.rca import rca, rpop, mcp
from .core.diversity import diversity, ubiquity, normalized_ubiquity
from .core.utils import (
    pivot_to_matrix,
    melt_matrix,
    binarize,
    normalize_zscore,
    normalize_01,
    make_sample_data,
)
from .core.preprocess import trim_core

# ── Complexity ─────────────────────────────────────────────────────────────────
from .complexity.eci_pci import eci_pci
from .complexity.eigenvector import eci_pci_eigenvector
from .complexity.reflections import method_of_reflections, mor_regions, mor_activities
from .complexity.fitness import fitness_complexity
from .complexity.subnational import subnational_eci

# ── Relatedness ────────────────────────────────────────────────────────────────
from .relatedness.proximity import (
    proximity,
    continuous_proximity,
    cosine_proximity,
    correlation_proximity,
    relatedness,
)
from .relatedness.density import (
    relatedness_density,
    density,
    distance,
    relatedness_density_internal,
    relatedness_density_external,
    relative_relatedness,
)
from .relatedness.cooccurrence import co_occurrence, relatedness_index, z_score_novelty
from .relatedness.cross_space import cross_proximity, cross_relatedness, cross_space_proximity

# ── Specialization ─────────────────────────────────────────────────────────────
from .specialization.location_quotient import (
    location_quotient,
    location_quotient_avg,
    hachman_index,
    specialization_coefficient,
    spec_coefficient,
    krugman_index,
)
from .specialization.similarity import export_similarity

# ── Inequality ─────────────────────────────────────────────────────────────────
from .inequality.gini import gini, locational_gini, hoover_gini
from .inequality.concentration import herfindahl, hhi, shannon_entropy, hoover_index

# ── Productivity ───────────────────────────────────────────────────────────────
from .productivity.prody import (
    prody,
    expy,
    product_gini_index,
    pgi,
    product_emissions_index,
    peii,
)

# ── Patents ────────────────────────────────────────────────────────────────────
from .patents.recombination import (
    ease_of_recombination,
    modular_complexity,
    modular_complexity_avg,
)

# ── Dynamics ───────────────────────────────────────────────────────────────────
from .dynamics.growth import growth_rate, growth_matrix, growth_rates
from .dynamics.entry_exit import (
    entry,
    exit,
    entry_exit_summary,
    entry_tracking,
    exit_tracking,
)

# ── Outlook ────────────────────────────────────────────────────────────────────
from .outlook.coi_cog import complexity_outlook_index, complexity_outlook_gain, coi, cog

# ── ECI Optimization (Stojkoski & Hidalgo 2026) ───────────────────────────────
from .optimization import (
    calibrate_steppingstone,
    effort_matrix,
    forecast_specialization,
    eci_optimization,
    calibrate_growth_model,
    expected_growth,
    eci_target_for_growth,
)

# ── Strategic diffusion (Alshamsi, Pinheiro & Hidalgo 2018) ───────────────────
from .optimization import (
    proximity_network,
    activation_probabilities,
    calibrate_contagion,
    diversification_strategy,
    expected_diversification_time,
    compare_strategies,
    optimize_sequence,
)

# ── High-level pipeline ────────────────────────────────────────────────────────
from .pipeline import compute_complexity

__all__ = [
    # core
    "rca", "rpop", "mcp",
    "diversity", "ubiquity", "normalized_ubiquity",
    "pivot_to_matrix", "melt_matrix", "binarize",
    "normalize_zscore", "normalize_01",
    "make_sample_data", "trim_core",
    # complexity
    "eci_pci", "eci_pci_eigenvector",
    "method_of_reflections", "mor_regions", "mor_activities",
    "fitness_complexity",
    "subnational_eci",
    # relatedness
    "proximity", "continuous_proximity",
    "cosine_proximity", "correlation_proximity",
    "relatedness_density", "density", "relatedness", "distance",
    "relatedness_density_internal", "relatedness_density_external",
    "relative_relatedness",
    "co_occurrence", "relatedness_index", "z_score_novelty",
    "cross_proximity", "cross_relatedness", "cross_space_proximity",
    # specialization
    "location_quotient", "location_quotient_avg",
    "hachman_index", "specialization_coefficient", "spec_coefficient",
    "krugman_index",
    "export_similarity",
    # inequality
    "gini", "locational_gini", "hoover_gini",
    "herfindahl", "hhi", "shannon_entropy", "hoover_index",
    # productivity
    "prody", "expy",
    "product_gini_index", "pgi",
    "product_emissions_index", "peii",
    # patents
    "ease_of_recombination", "modular_complexity", "modular_complexity_avg",
    # dynamics
    "growth_rate", "growth_matrix", "growth_rates",
    "entry", "exit", "entry_exit_summary",
    "entry_tracking", "exit_tracking",
    # outlook
    "complexity_outlook_index", "complexity_outlook_gain", "coi", "cog",
    # optimization
    "calibrate_steppingstone", "effort_matrix", "forecast_specialization",
    "eci_optimization",
    "calibrate_growth_model", "expected_growth", "eci_target_for_growth",
    # strategic diffusion
    "proximity_network", "activation_probabilities", "calibrate_contagion",
    "diversification_strategy", "expected_diversification_time",
    "compare_strategies", "optimize_sequence",
    # pipeline
    "compute_complexity",
]
