# econcomplex

`econcomplex` is a Python library for economic complexity and regional science indicators. It consolidates tools for RCA, ECI, PCI, relatedness density, specialization, inequality, productivity, dynamics, and complexity outlook analysis in a single package.

## Highlights

- Economic complexity indicators with a single entry point — `eci_pci(mat, method="eigenvector" | "reflections" | "fitness")` — plus subnational ECI
- Automatic pre-processing of sparse data: degenerate units (zero diversity/ubiquity) are trimmed and returned as `NaN` (`trim_core`)
- Relatedness and product-space measures such as proximity (discrete, continuous, cosine, correlation), co-occurrence, density, and relative relatedness
- Regional specialization metrics such as location quotient, Hachman, Krugman, and export similarity
- Inequality and concentration measures such as Gini, Hoover, Herfindahl, and Shannon entropy
- Productivity and dynamics indicators such as PRODY, EXPY, growth, and entry/exit analysis (matrix and panel APIs)
- **ECI Optimization** (Stojkoski & Hidalgo 2026): minimal-effort diversification portfolios that reach an ECI or growth target
- **Strategic diffusion** (Alshamsi, Pinheiro & Hidalgo 2018): complex-contagion model, diversification strategies, and optimal sequencing on the product space

## Installation

```bash
pip install git+https://github.com/eltonfreitas/econcomplex.git
```

For editable local development:

```bash
pip install -e .[dev]
```

## Quick Start

```python
import pandas as pd
import econcomplex as ec

df = pd.read_csv("my_data.csv")

result = ec.compute_complexity(
    df,
    cols={"loc": "region", "act": "sector", "val": "employment"},
    method="eigenvector",
    compute_coi_cog=False,
)

print(result.head())
```

You can also work directly with matrices:

```python
mat = ec.pivot_to_matrix(df, "region", "sector", "employment")
rca = ec.rca(mat)
eci, pci = ec.eci_pci(mat)                      # method="reflections"/"fitness" also available
phi = ec.proximity(mat)["product"]
density = ec.density(mat, phi=phi)
```

And identify diversification targets with the optimization layer
(requires a panel with at least the periods t, t+5 and t+10):

```python
model = ec.calibrate_steppingstone(panel, "region", "sector", "employment", "year",
                                   horizon=10, steppingstone=5)
portfolio = ec.eci_optimization(mat, model, delta_eci=0.1)
```

## Documentation

- English technical documentation PDF: [docs/econcomplex_documentation_en.pdf](docs/econcomplex_documentation_en.pdf)
- Portuguese technical documentation PDF: [docs/econcomplex_documentation_pt.pdf](docs/econcomplex_documentation_pt.pdf)
- English LaTeX source: [docs/econcomplex_documentation_en.tex](docs/econcomplex_documentation_en.tex)
- Portuguese LaTeX source: [docs/econcomplex_documentation_pt.tex](docs/econcomplex_documentation_pt.tex)
- Portuguese project overview: [README.pt-BR.md](README.pt-BR.md)

## Project Structure

- `econcomplex/`: importable package
- `tests/`: automated tests
- `examples/`: runnable usage examples (`python examples/basic_usage.py`)
- `docs/`: English and Portuguese documentation sources and PDFs

## Testing

```bash
pytest
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

