# Changelog

## [1.4.0] — 2026-11-25

### Added
- `pca_weights`: weights for `desirability_index` from principal component
  analysis, optionally within dimensions of equal weight — the device of the
  smart diversification score of Romero et al. (2025).
- `complexity_outlook_gain(net=True)`: subtracts the term of the activity
  itself and returns the exact change in the Complexity Outlook Index.
- `explain_entry(mode="vector")`: for the models that predict entry from the
  whole RCA vector, which present activities make the entry into each target
  more likely. The importance tables gain a `direction` column.

### Changed
- `io_relatedness`: the default is now confirmed against eq. (1) of the
  published Essletzbichler (2015), not only the working paper.
- Portuguese documentation: section titles in sentence case.

## [1.3.0] — 2026-11-25

### Added
- `full_relatedness`: relatedness density with continuous specialization
  (Cicerone, McCann & Venhorst 2026). Every related activity is weighted by
  how specialized the region is in it, instead of counting only those with
  RCA >= 1, and the index splits exactly into the part that comes from
  specialized activities and the part that comes from non-specialized ones
  (`part="specialized"` / `"non_specialized"`). Weights: the symmetric index
  of Iapadre (2001), as in the paper, the one of Dalum et al. (1998), or a
  capped RCA, whose specialized part is the standard density.
- `rsca`: revealed symmetric comparative advantage, in [-1, 1]
  (`method="dalum"` or `"iapadre"`).

### Fixed
- Documentation: the note on the corrected COG said "up to version 1.2.0";
  the old normalization was in place up to 1.1.0.

## [1.2.0] — 2026-11-25

General-purpose tools for the steps that come after the indicators:
explaining opportunities, ranking them, reading them through lists of
activities, and testing whether a score predicts entry. 112 public
functions (22 new). scikit-learn becomes an optional dependency
(`pip install "econcomplex[ml]"`), needed only by the prediction models.

### Fixed
- **`complexity_outlook_gain` (`cog`) used the wrong normalization.** The
  gain from entering activity c was divided by the total proximity of c,
  instead of dividing each term by the total proximity of the activity c'
  whose density rises. The two only coincide when every activity has the
  same total proximity. The function now equals the actual change in the
  Complexity Outlook Index when the activity is added (tested), and matches
  the Growth Lab's `py-ecomplexity`. **Values change**: on the sample data
  the correlation with the previous output is about 0.97, so rankings of
  activities by COG can move. `compute_complexity` returns the corrected
  `cog` column; `coi` is unchanged.

### Added
- **Complexity**
  - `genepy`, also `eci_pci(mat, method="genepy")`: the GENEPY index of
    Sciarra et al. (2020), following the authors' released code.
  - `eci_diagnostics`: spectral gap, relation with diversity and stability
    of the ECI ranking under resampling of activities (Mealy, Farmer &
    Teytelboym 2019). Flags matrices where the sign convention of the ECI
    is fragile.
- **Relatedness**
  - `implied_rca`: implied comparative advantage of Hausmann, Stock &
    Yıldırım (2022) — RCA predicted from the k most similar activities or
    regions.
  - `io_relatedness`: proximity from an input-output table (Essletzbichler
    2015; Fan & Lang 2000).
  - `skill_relatedness`: relatedness from a matrix of flows between
    activities (job switches), observed over expected, mapped to [-1, 1)
    (Neffke, Otto & Weyh 2017).
  - `spatial_density`: share of a region's neighbours specialized in each
    activity, with user-supplied spatial weights (Bahar, Hausmann & Hidalgo
    2014; Boschma, Martín & Minondo 2017).
  - `density_contributions`: which of the region's activities make another
    activity feasible; the contributions add up to the density.
  - `decomposed_density`: density net of region and activity fixed effects
    (handles NaN cells, e.g. only the activities a region lacks).
- **Strategy** (new `strategy` module)
  - `desirability_index`: weighted combination of standardized indicators
    per activity, with explicit (possibly negative) weights.
  - `weight_sensitivity`: how far each activity's rank moves when the
    weights are redrawn (Dirichlet).
  - `strategy_quadrants`: short jump / long bet / low value / avoid, with
    cuts computed inside each region.
  - `opportunity_alignment`: correlation, per region, between the
    feasibility and the desirability of its options.
- **Policy lists** (new `policy` module) — for any set of activities (green
  lists, missions, value chains)
  - `list_complexity_potential`: presence and potential of each region in
    each list; generalizes the Green Complexity Index and Potential of
    Mealy & Teytelboym (2022), reproduced with `average="missing"`.
  - `list_density`: part of the relatedness density that rests on the
    activities of a list (additive).
  - `missing_links`: links of a chain that are present, feasible or missing
    in each region.
  - `transition_risk`: exposure to a list of activities at risk, Brown
    Lock-in Index and Transition Outlook (Andres et al. 2023).
- **Entry prediction** (new `prediction` module)
  - `entry_targets`: binary entry target and risk set between two periods
    (RCA < 0.5 → RCA ≥ 1 by default, optional minimum size).
  - `validate_entry`: scores any predictor of entry — AUC, within-region
    AUC, average precision, precision/recall@k, Brier — with bootstrap
    intervals by region and differences to a baseline. NumPy/SciPy only.
  - `build_features`, `predict_entry`, `explain_entry` (need scikit-learn):
    table of features per cell, out-of-sample probability of entry by
    cross-validation over regions (gradient boosting, random forest, logit
    or any classifier; one model on layers or one model per activity on the
    RCA vector), and permutation importance on held-out regions.
  - `potential_employment`: expected size of entries, probability × arrival
    size (location quotient of 1, or quantiles of past entrants).
- `examples/diversification_strategy.py`: the new tools end to end.

### Changed
- Documentation: a step-by-step tutorial opens the reference part of both
  PDFs — nine steps from the raw table to a shortlist of targets and its
  out-of-sample test, each with code, the output it prints and how to read
  it. It is generated by `docs/generate_tutorial.py`, which runs the code,
  so the numbers in the PDF are the ones the library produces; the same
  code is in `examples/tutorial.py`.
- Documentation: new sections in the English and Portuguese PDFs (density
  decompositions, skill relatedness, spatial density, strategy, policy
  lists, entry validation), COG formula, interpretation table, references.
- `econcomplex.relatedness.__all__` lists every function of the subpackage.
- CI installs the `ml` extra so that the prediction tests run.

## [1.1.0] — 2026-11-24

### Fixed
- **Label alignment.** Auxiliary inputs given as labelled pandas objects are
  now matched to the matrix by label instead of by position. Before, a
  `Series` or `DataFrame` with the same labels in a different order silently
  produced wrong results. Affects `prody`, `expy`, `pgi`, `peii`, `rpop`,
  `hoover_index`, `hoover_gini`, `coi`, `cog`, every density function (`phi`),
  `cross_relatedness` (`x_phi`), `subnational_eci`, `modular_complexity`
  (`eor`) and the strategic-diffusion functions (`active`). Lists, arrays and
  Series with a default integer index are still taken positionally; a missing
  label raises a `ValueError` naming it.
- **Input validation.** Matrices with NaN/infinite or negative cells now raise
  a clear `ValueError` at the entry point. NaN used to surface as a misleading
  "trimming to the (1, 1)-core" error and negative values were accepted.
- **`eci_optimization` no longer stalls.** The exact solver could run
  indefinitely on locations with many near-equivalent candidates. It is now
  capped by `time_limit` (default 5 s per location); when the cap is hit the
  best feasible portfolio found is returned, never worse than the greedy one.
- `subnational_eci` returns NaN (it returned 0) for regions with no
  specialization, and accepts an external PCI in any order, with extra
  activities, or missing some of the matrix's activities.
- Citations: Catalán, Navarrete & Figueroa (2022) for the cross-space, and
  Pinheiro et al. (2022) for relative relatedness.

### Added
- `product_proximity` and `location_proximity`: return the proximity matrix
  directly instead of the `{"product": ..., "location": ...}` dictionary.
- `percent` option in `relatedness_density` and its internal/external
  variants: `percent=False` returns shares (0-1), the scale of `distance`
  and `cross_relatedness`. The default (0-100) is unchanged.
- `exits`, the preferred name for `exit` (which shadows the Python builtin).
- Runnable `Examples` in the docstrings of the main entry points, executed
  by the test suite (`pytest --doctest-modules`).
- Continuous integration (GitHub Actions): Python 3.10 to 3.14 and NumPy 1.x.
- `eci_optimization` output has an `optimal` column (True when the solver
  proved optimality) and warns once when locations reach the time limit.
- Known-answer tests for the 25 public functions that had no direct test,
  plus a regression test that reproduces the 1.0.1 results to 1e-7.

### Changed
- Minimum Python is now 3.10. Python 3.9 reached end of life in October 2025
  and is no longer available on the CI runners; `pip` on 3.9 keeps
  installing 1.0.1.

### Note
- The 1.0.1 entry below says outputs were "byte-for-byte identical" to 1.0.0;
  the check actually used a tolerance of 1e-9.

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
