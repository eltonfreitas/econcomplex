# econcomplex

[![version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)](tests/)

**econcomplex** is a Python library for **economic complexity and regional
science indicators**. It consolidates, in a single coherent API, the tools
scattered across the reference packages of the field — `EconGeo` (R),
`economiccomplexity` (R), `py-ecomplexity`, `py-economic-complexity` — and
adds a **target-oriented optimization layer** (ECI Optimization and strategic
diffusion) that, to our knowledge, is not available in any other package.

*Leia em português: [README.pt-BR.md](README.pt-BR.md).*

---

## What it computes

| Group | Indicators |
|---|---|
| **Complexity** | ECI / PCI through a single entry point — `eci_pci(mat, method="eigenvector" \| "reflections" \| "fitness")` — plus subnational ECI projected with an external PCI |
| **Relatedness / Product Space** | Proximity (discrete, correlation, cosine), relatedness density, distance, relative relatedness (option-set z-score), co-occurrence indices, cross-space proximity between two activity spaces |
| **Specialization** | Location quotient, Hachman, Krugman, Hoover specialization coefficient, export similarity |
| **Inequality / Concentration** | Gini, locational Gini, Hoover-Gini, Hoover index, Herfindahl-Hirschman, Shannon entropy |
| **Productivity** | PRODY, EXPY, Product Gini Index, Product Emissions Intensity Index |
| **Patents** | Ease of recombination, modular complexity |
| **Dynamics** | Growth rates, entry/exit tracking — matrix-pair and long-panel APIs |
| **Outlook** | Complexity Outlook Index (COI) and Gain (COG) |
| **ECI Optimization** | Stepping-stone forecast model, entry-effort matrix, exact 0–1 program for minimal-effort diversification portfolios, growth targeting (Stojkoski & Hidalgo 2026) |
| **Strategic diffusion** | Complex-contagion calibration, five diversification strategies, optimal entry sequencing (Alshamsi, Pinheiro & Hidalgo 2018) |

87 public functions in total — the PDF documentation carries a complete API
reference and an interpretation guide for every indicator family.

## Installation

```bash
pip install git+https://github.com/eltonfreitas/econcomplex.git
```

Requires Python ≥ 3.9 with `numpy ≥ 1.21` (1.x **and** 2.x supported),
`pandas ≥ 1.3`, `scipy ≥ 1.9`. For local development:

```bash
git clone https://github.com/eltonfreitas/econcomplex.git
cd econcomplex
pip install -e .[dev]
pytest          # 81 tests
```

## Quick start

### 1. One call, every indicator (long-format data)

```python
import pandas as pd
import econcomplex as ec

df = pd.read_csv("my_data.csv")        # columns: region, sector, employment[, year]

result = ec.compute_complexity(
    df,
    cols={"loc": "region", "act": "sector", "val": "employment", "time": "year"},
    method="eigenvector",              # or "reflections" / "fitness"
)
# adds columns: rca, mcp, diversity, ubiquity, eci, pci, density, distance, coi, cog
# with a "time" column the pipeline recomputes everything per period automatically
```

### 2. Working with matrices

```python
mat = ec.pivot_to_matrix(df, "region", "sector", "employment")

eci, pci   = ec.eci_pci(mat)                      # eigenvector method (default)
eci2, pci2 = ec.eci_pci(mat, method="fitness")    # same call, other method

phi     = ec.proximity(mat)["product"]            # product space
density = ec.density(mat, phi=phi)                # 0–100 % relatedness density
coi     = ec.coi(mat, pci, phi=phi)               # diversification potential
```

Degenerate units (zero diversity or ubiquity) are **trimmed automatically**
and returned as `NaN`; for very sparse data (e.g. municipal trade) use the
well-connected core: `ec.eci_pci(mat, dmin=2, umin=2)` or `ec.trim_core(mat, 2, 2)`.

### 3. Diversification targets (ECI Optimization)

Requires a panel with at least the periods *t*, *t+τ* and *t+Δt*:

```python
model = ec.calibrate_steppingstone(panel, "region", "sector", "employment",
                                   "year", horizon=10, steppingstone=5)

portfolio = ec.eci_optimization(mat, model, delta_eci=0.1)
# → minimal-effort set of new activities per region that raises its ECI by 0.1

# Growth targeting: convert a 3.5 %/yr target into an ECI target
gm       = ec.calibrate_growth_model(macro, "region", "year", "gdppc", "eci")
eci_star = ec.eci_target_for_growth(gm, 0.035, gdppc_now)
portfolio = ec.eci_optimization(mat, model, target_eci=eci_star)

# When to make unrelated bets (strategic diffusion)
adj  = ec.proximity_network(mat)
fit  = ec.calibrate_contagion(panel, "region", "sector", "employment", "year",
                              adjacency=adj)
best = ec.optimize_sequence(adj, ec.mcp(mat).loc["my_region"],
                            B=fit["B"], alpha=fit["alpha"])
```

## Data format

The high-level API expects **long-format** (tidy) data — one row per
(location, activity[, period]):

| region | sector | employment | year |
|---|---|---:|---|
| SP | cnae_10 | 12345 | 2022 |
| SP | cnae_25 | 6789 | 2022 |
| RJ | cnae_10 | 9012 | 2022 |

Requirements: no duplicate (location, activity, period) rows, non-negative
values, no `NaN`, a single geographic level and a single activity
classification per analysis. Works with employment, exports, patents,
payroll — anything shaped location × activity × value. To experiment without
data: `df = ec.make_sample_data(n_locs=50, n_acts=30, seed=42)`.

## Documentation and examples

- **Technical documentation (PDF)** — formulas, step-by-step usage,
  interpretation guide, and the complete API reference:
  [English](docs/econcomplex_documentation_en.pdf) ·
  [Português](docs/econcomplex_documentation_pt.pdf)
  (LaTeX sources in [docs/](docs/))
- **Runnable examples**: [examples/basic_usage.py](examples/basic_usage.py)
  (guided tour of every indicator group) and
  [examples/eci_optimization.py](examples/eci_optimization.py)
  (optimization layer end to end)
- **In-code reference**: every function has a full NumPy-style docstring —
  `help(ec.eci_pci)`
- **[CHANGELOG.md](CHANGELOG.md)** — release history

The API has three layers (detailed map in the PDF): *entry points* such as
`eci_pci` and `compute_complexity`; *advanced implementations* they delegate
to (`method_of_reflections`, `fitness_complexity`, …); and short *aliases*
bound to the same objects (`density`, `hhi`, `coi`, `pgi`, …).

## Validation

The 81-test suite includes exact validations against the literature: the
eigenvector ECI/PCI uses the proper non-symmetric solver; the strategic
diffusion module reproduces the closed-form solution of Alshamsi et al.
(2018, eq. 2) on the wheel network; relative relatedness follows Pinheiro
et al. (2022, eq. 7) exactly; and the 0–1 portfolio program is solved
exactly with `scipy.optimize.milp`. On the 2022–2024 BACI trade data the
library recovers the canonical ECI country ranking.

## Citation

```bibtex
@software{freitas_econcomplex_2026,
  author  = {Freitas, Elton},
  title   = {econcomplex: economic complexity and regional science indicators in Python},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/eltonfreitas/econcomplex}
}
```

Please also cite the original papers of the indicators you use — full list
in the PDF documentation. Key references: Hidalgo & Hausmann (2009, *PNAS*);
Hidalgo et al. (2007, *Science*); Tacchella et al. (2012, *Sci. Rep.*);
Alshamsi, Pinheiro & Hidalgo (2018, *Nat. Commun.*); Pinheiro et al. (2022,
*Res. Policy*); Stojkoski & Hidalgo (2026, *Res. Policy*).

## License

MIT — see [LICENSE](LICENSE).
