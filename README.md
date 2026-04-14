# econcomplex

`econcomplex` is a Python library for economic complexity and regional science indicators. It consolidates tools for RCA, ECI, PCI, relatedness density, specialization, inequality, productivity, dynamics, and complexity outlook analysis in a single package.

## Highlights

- Economic complexity indicators such as ECI, PCI, method of reflections, and fitness complexity
- Relatedness and product-space measures such as proximity, co-occurrence, and density
- Regional specialization metrics such as location quotient, Hachman, Krugman, and export similarity
- Inequality and concentration measures such as Gini, Hoover, Herfindahl, and Shannon entropy
- Productivity and dynamics indicators such as PRODY, EXPY, growth, and entry/exit analysis

## Installation

```bash
pip install .
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
eci, pci = ec.eci_pci(mat)
phi = ec.proximity(mat)["product"]
density = ec.relatedness_density(mat, phi=phi)
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
- `example_usage.py`: usage examples
- `docs/`: English and Portuguese documentation sources and PDFs

## Testing

```bash
pytest
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

