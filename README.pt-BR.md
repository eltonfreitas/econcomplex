# econcomplex

`econcomplex` e uma biblioteca Python para indicadores de complexidade economica e ciencia regional. Ela consolida ferramentas para RCA, ECI, PCI, densidade de relatedness, especializacao, desigualdade, produtividade, dinamica e complexity outlook em um unico pacote.

## Destaques

- Indicadores de complexidade economica com porta de entrada unica — `eci_pci(mat, method="eigenvector" | "reflections" | "fitness")` — alem de ECI subnacional
- Pre-processamento automatico de dados esparsos: unidades degeneradas (diversidade/ubiquidade zero) sao removidas e retornadas como `NaN` (`trim_core`)
- Medidas de relatedness e product space como proximidade (discreta, continua, cosseno, correlacao), coocorrencia, densidade e relatedness relativa
- Indicadores de especializacao regional como location quotient, Hachman, Krugman e similaridade de exportacao
- Medidas de desigualdade e concentracao como Gini, Hoover, Herfindahl e entropia de Shannon
- Indicadores de produtividade e dinamica como PRODY, EXPY, crescimento e entrada/saida (APIs de matriz e painel)
- **Otimizacao de ECI** (Stojkoski & Hidalgo 2026): portfolios de diversificacao de menor esforco para atingir uma meta de ECI ou de crescimento
- **Difusao estrategica** (Alshamsi, Pinheiro & Hidalgo 2018): modelo de contagio complexo, estrategias de diversificacao e sequenciamento otimo no product space

## Instalacao

```bash
pip install git+https://github.com/eltonfreitas/econcomplex.git
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

Tambem e possivel trabalhar diretamente com matrizes e usar a camada de
otimizacao (exige painel com ao menos os periodos t, t+5 e t+10):

```python
mat = ec.pivot_to_matrix(df, "regiao", "setor", "emprego")
eci, pci = ec.eci_pci(mat)        # method="reflections"/"fitness" tambem

model = ec.calibrate_steppingstone(painel, "regiao", "setor", "emprego", "ano",
                                   horizon=10, steppingstone=5)
portfolio = ec.eci_optimization(mat, model, delta_eci=0.1)
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
- `examples/`: exemplos de uso executaveis (`python examples/basic_usage.py`)
- `docs/`: documentacao em ingles e portugues

## Testes

```bash
pytest
```

## Licenca

Este projeto esta licenciado sob a licenca MIT. Veja [LICENSE](LICENSE).

