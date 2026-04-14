# econcomplex

`econcomplex` e uma biblioteca Python para indicadores de complexidade economica e ciencia regional. Ela consolida ferramentas para RCA, ECI, PCI, densidade de relatedness, especializacao, desigualdade, produtividade, dinamica e complexity outlook em um unico pacote.

## Destaques

- Indicadores de complexidade economica como ECI, PCI, metodo das reflexoes e fitness complexity
- Medidas de relatedness e product space como proximidade, coocorrencia e densidade
- Indicadores de especializacao regional como location quotient, Hachman, Krugman e similaridade de exportacao
- Medidas de desigualdade e concentracao como Gini, Hoover, Herfindahl e entropia de Shannon
- Indicadores de produtividade e dinamica como PRODY, EXPY, crescimento e entrada/saida

## Instalacao

```bash
pip install .
```

Para desenvolvimento local editavel:

```bash
pip install -e .[dev]
```

## Exemplo Rapido

```python
import pandas as pd
import econcomplex as ec

df = pd.read_csv("meus_dados.csv")

result = ec.compute_complexity(
    df,
    cols={"loc": "regiao", "act": "setor", "val": "emprego"},
    method="eigenvector",
    compute_coi_cog=False,
)

print(result.head())
```

## Documentacao

- PDF em ingles: [docs/econcomplex_documentation_en.pdf](docs/econcomplex_documentation_en.pdf)
- PDF em portugues: [docs/econcomplex_documentation_pt.pdf](docs/econcomplex_documentation_pt.pdf)
- Fonte LaTeX em ingles: [docs/econcomplex_documentation_en.tex](docs/econcomplex_documentation_en.tex)
- Fonte LaTeX em portugues: [docs/econcomplex_documentation_pt.tex](docs/econcomplex_documentation_pt.tex)
- Visao geral em ingles: [README.md](README.md)

## Estrutura do Projeto

- `econcomplex/`: pacote Python importavel
- `tests/`: testes automatizados
- `example_usage.py`: exemplos de uso
- `docs/`: documentacao em ingles e portugues

## Testes

```bash
pytest
```

## Licenca

Este projeto esta licenciado sob a licenca MIT. Veja [LICENSE](LICENSE).

