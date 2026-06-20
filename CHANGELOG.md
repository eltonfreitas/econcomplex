# Changelog

## [1.0.1] — 2026-06-13

### Changed
- `compute_complexity` is now a **pure orchestration layer**: it calls the
  same public functions a user would call by hand (`diversity`, `ubiquity`,
  and `distance` are now delegated instead of recomputed inline), so any
  change to an indicator propagates automatically. Outputs are byte-for-byte
  identical to 1.0.0 (verified against a saved baseline for all three methods).

### Added
- `compute_complexity` now exposes `trim`, `dmin`, and `umin`, forwarded to
  `eci_pci` — the well-connected `(2, 2)` core (recommended for sparse
  subnational data) is finally reachable from the high-level pipeline, not
  only from the matrix-level API.

## [1.0.0] — 2026-06-12

First official release.

### Added
- **Single entry point for complexity**: `eci_pci(mat, method="eigenvector" | "reflections" | "fitness")` in its own module (`complexity/eci_pci.py`), mirroring the interface of the R `economiccomplexity` package. Method-specific implementations remain public for advanced use (`eci_pci_eigenvector`, `method_of_reflections`, `fitness_complexity`).
- **Pre-processing for sparse data**: `trim_core(mat, dmin, umin)` iteratively prunes degenerate units (zero diversity/ubiquity) recomputing RCA each pass; applied automatically by `eci_pci` (`trim=True`, trimmed units returned as `NaN`). Use `dmin=2, umin=2` for the well-connected core recommended for subnational data.
- **ECI Optimization** (Stojkoski & Hidalgo 2026, *Research Policy* 55:105454): `calibrate_steppingstone`, `effort_matrix`, `forecast_specialization`, and `eci_optimization` (exact 0–1 program via `scipy.optimize.milp`), plus growth targeting (`calibrate_growth_model`, `eci_target_for_growth`, `expected_growth`).
- **Strategic diffusion** (Alshamsi, Pinheiro & Hidalgo 2018, *Nat. Commun.* 9:1328): `proximity_network`, `calibrate_contagion`, `activation_probabilities`, `diversification_strategy` (5 strategies), `expected_diversification_time` (validated against the paper's closed-form eq. 2), `compare_strategies`, and `optimize_sequence` (simulated annealing).
- **Documented short API**: aliases bound to the canonical functions (`density`, `relatedness`, `hhi`, `coi`, `cog`, `pgi`, `peii`, `spec_coefficient`, `cross_space_proximity`), new functions `make_sample_data`, `cosine_proximity`, `correlation_proximity`, and long-format panel wrappers `growth_rates`, `entry_tracking`, `exit_tracking`.
- `log_fitness` option (Cristelli et al. 2015 log scale) and convergence `tol` for the Method of Reflections; non-convergence warning for Fitness-Complexity.
- `continuous_method="correlation" | "cosine"` in `proximity()`/`continuous_proximity()`.
- Runnable examples in `examples/` (`basic_usage.py`, `eci_optimization.py`) and an API map section in the documentation.
- Expanded documentation: complete auto-generated API reference (87 functions, `docs/generate_api_reference.py`), indicator interpretation guide, and rewritten bilingual READMEs with quickstarts, data format, validation notes, and BibTeX citation.

### Changed
- Default iterations unified at **20** for reflections and fitness (matching the R `economiccomplexity` package; for fitness it is a cap with early stopping at convergence).
- Method of Reflections output is now sign-oriented like the eigenvector method (ECI correlates positively with diversity, PCI negatively with ubiquity).
- `compute_complexity` routes all methods through the `eci_pci` dispatcher (validation and trimming included).
- Internal relative relatedness of the optimization module now follows Pinheiro et al. (2022, eq. 7) exactly (z-transform over the option set).
- SciPy minimum raised to 1.9 (`scipy.optimize.milp`).
- Documentation (EN/PT) fully revised: GitHub installation, real API signatures, new sections (pre-processing, ECI Optimization, strategic diffusion), corrected references (Tacchella et al. 2012).

### Fixed
- NumPy 2.x compatibility: `np.trapz` removal broke `locational_gini`/`hoover_gini`.
- `cross_relatedness` returned the wrong column labels (crashed whenever the two spaces had different sizes).
- Missing standard-deviation guards in the eigenvector sign correction (silent NaN on degenerate matrices).
- Edge-case guards in the optimization layer (`b1 ≈ 0` effort, `a1 + a3·z ≈ 0` growth-target inversion).
- Documentation/API mismatches reported by external testing (`rpop`, `pgi`, `peii`, `expy` signatures; `coi`/`cog` argument order; `proximity` return type).

## [0.1.0] — 2026 (initial public release)

- Core indicators: RCA, RPOP, Mcp, diversity/ubiquity; ECI/PCI (eigenvector, reflections, fitness); proximity, relatedness density, co-occurrence, cross-space; specialization, inequality, productivity, patents, dynamics, COI/COG; `compute_complexity` pipeline.
