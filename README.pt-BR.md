# econcomplex

[![version](https://img.shields.io/badge/vers%C3%A3o-1.0.0-blue)](CHANGELOG.md)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/licen%C3%A7a-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/testes-81%20passando-brightgreen)](tests/)

**econcomplex** é uma biblioteca Python para **indicadores de complexidade
econômica e ciência regional**. Ela consolida, numa API única e coerente, as
ferramentas espalhadas pelos pacotes de referência da área — `EconGeo` (R),
`economiccomplexity` (R), `py-ecomplexity`, `py-economic-complexity` — e
adiciona uma **camada de otimização orientada a metas** (Otimização de ECI e
difusão estratégica) que, até onde sabemos, não existe em nenhum outro pacote.

*Read in English: [README.md](README.md).*

---

## O que ela calcula

| Grupo | Indicadores |
|---|---|
| **Complexidade** | ECI / PCI por uma porta de entrada única — `eci_pci(mat, method="eigenvector" \| "reflections" \| "fitness")` — além de ECI subnacional projetado com PCI externo |
| **Relatedness / Product Space** | Proximidade (discreta, correlação, cosseno), densidade de relatedness, distância, relatedness relativa (z-score no option set), índices de coocorrência, proximidade cross-space entre dois espaços de atividades |
| **Especialização** | Quociente locacional, Hachman, Krugman, coeficiente de especialização de Hoover, similaridade de exportações |
| **Desigualdade / Concentração** | Gini, Gini locacional, Hoover-Gini, índice de Hoover, Herfindahl-Hirschman, entropia de Shannon |
| **Produtividade** | PRODY, EXPY, Product Gini Index, Product Emissions Intensity Index |
| **Patentes** | Facilidade de recombinação, complexidade modular |
| **Dinâmica** | Taxas de crescimento, rastreamento de entrada/saída — APIs por pares de matrizes e por painel longo |
| **Outlook** | Complexity Outlook Index (COI) e Gain (COG) |
| **Otimização de ECI** | Modelo de previsão com steppingstone, matriz de esforço de entrada, programa 0–1 exato para portfólios de diversificação de menor esforço, metas de crescimento (Stojkoski & Hidalgo 2026) |
| **Difusão estratégica** | Calibração de contágio complexo, cinco estratégias de diversificação, sequenciamento ótimo de entrada (Alshamsi, Pinheiro & Hidalgo 2018) |

São 87 funções públicas — a documentação em PDF traz a referência completa da
API e um guia de interpretação para cada família de indicadores.

## Instalação

```bash
pip install econcomplex
```

Ou, para a versão de desenvolvimento mais recente direto do GitHub:

```bash
pip install git+https://github.com/eltonfreitas/econcomplex.git
```

Exige Python ≥ 3.9 com `numpy ≥ 1.21` (compatível com 1.x **e** 2.x),
`pandas ≥ 1.3`, `scipy ≥ 1.9`. Para desenvolvimento local:

```bash
git clone https://github.com/eltonfreitas/econcomplex.git
cd econcomplex
pip install -e .[dev]
pytest          # 81 testes
```

## Começando

### 1. Uma chamada, todos os indicadores (dados em formato longo)

```python
import pandas as pd
import econcomplex as ec

df = pd.read_csv("meus_dados.csv")     # colunas: regiao, setor, emprego[, ano]

resultado = ec.compute_complexity(
    df,
    cols={"loc": "regiao", "act": "setor", "val": "emprego", "time": "ano"},
    method="eigenvector",              # ou "reflections" / "fitness"
)
# adiciona as colunas: rca, mcp, diversity, ubiquity, eci, pci, density,
# distance, coi, cog — com a coluna de tempo, recalcula tudo por período
```

### 2. Trabalhando com matrizes

```python
mat = ec.pivot_to_matrix(df, "regiao", "setor", "emprego")

eci, pci   = ec.eci_pci(mat)                      # método do autovetor (padrão)
eci2, pci2 = ec.eci_pci(mat, method="fitness")    # mesma chamada, outro método

phi     = ec.proximity(mat)["product"]            # product space
density = ec.density(mat, phi=phi)                # densidade de relatedness (0–100 %)
coi     = ec.coi(mat, pci, phi=phi)               # potencial de diversificação
```

Unidades degeneradas (diversidade ou ubiquidade zero) são **podadas
automaticamente** e retornadas como `NaN`; para dados muito esparsos (ex.:
comércio municipal) use o núcleo conectado: `ec.eci_pci(mat, dmin=2, umin=2)`
ou `ec.trim_core(mat, 2, 2)`.

### 3. Alvos de diversificação (Otimização de ECI)

Exige um painel com ao menos os períodos *t*, *t+τ* e *t+Δt*:

```python
model = ec.calibrate_steppingstone(painel, "regiao", "setor", "emprego",
                                   "ano", horizon=10, steppingstone=5)

portfolio = ec.eci_optimization(mat, model, delta_eci=0.1)
# → menor conjunto de esforço de novas atividades, por região, que eleva o ECI em 0.1

# Meta de crescimento: converter 3,5 %/ano em alvo de ECI
gm       = ec.calibrate_growth_model(macro, "regiao", "ano", "pibpc", "eci")
eci_alvo = ec.eci_target_for_growth(gm, 0.035, pibpc_atual)
portfolio = ec.eci_optimization(mat, model, target_eci=eci_alvo)

# Quando fazer a aposta não-relacionada (difusão estratégica)
adj   = ec.proximity_network(mat)
fit   = ec.calibrate_contagion(painel, "regiao", "setor", "emprego", "ano",
                               adjacency=adj)
otimo = ec.optimize_sequence(adj, ec.mcp(mat).loc["minha_regiao"],
                             B=fit["B"], alpha=fit["alpha"])
```

## Formato dos dados

A API de alto nível espera dados em **formato longo** (tidy) — uma linha por
(local, atividade[, período]):

| regiao | setor | emprego | ano |
|---|---|---:|---|
| SP | cnae_10 | 12345 | 2022 |
| SP | cnae_25 | 6789 | 2022 |
| RJ | cnae_10 | 9012 | 2022 |

Requisitos: sem linhas duplicadas de (local, atividade, período), valores não
negativos, sem `NaN`, um único nível geográfico e uma única classificação de
atividades por análise. Funciona com emprego, exportações, patentes, massa
salarial — qualquer dado no formato local × atividade × valor. Para
experimentar sem dados: `df = ec.make_sample_data(n_locs=50, n_acts=30, seed=42)`.

## Documentação e exemplos

- **Documentação técnica (PDF)** — fórmulas, passo a passo de uso, guia de
  interpretação e a referência completa da API:
  [Português](docs/econcomplex_documentation_pt.pdf) ·
  [English](docs/econcomplex_documentation_en.pdf)
  (fontes LaTeX em [docs/](docs/))
- **Exemplos executáveis**: [examples/basic_usage.py](examples/basic_usage.py)
  (tour guiado por todos os grupos de indicadores) e
  [examples/eci_optimization.py](examples/eci_optimization.py)
  (camada de otimização de ponta a ponta)
- **Referência no código**: toda função tem docstring completo no estilo
  NumPy — `help(ec.eci_pci)`
- **[CHANGELOG.md](CHANGELOG.md)** — histórico de versões

A API tem três camadas (mapa detalhado no PDF): *portas de entrada* como
`eci_pci` e `compute_complexity`; *implementações avançadas* para as quais
elas delegam (`method_of_reflections`, `fitness_complexity`, …); e *aliases*
curtos ligados aos mesmos objetos (`density`, `hhi`, `coi`, `pgi`, …).

## Validação

A suíte de 81 testes inclui validações exatas contra a literatura: o ECI/PCI
por autovetor usa o solver não-simétrico correto; o módulo de difusão
estratégica reproduz a solução fechada de Alshamsi et al. (2018, eq. 2) na
rede wheel; a relatedness relativa segue Pinheiro et al. (2022, eq. 7)
exatamente; e o programa 0–1 de portfólio é resolvido de forma exata com
`scipy.optimize.milp`. Nos dados de comércio BACI 2022–2024 a biblioteca
recupera o ranking canônico de ECI dos países.

## Citação

```bibtex
@software{freitas_econcomplex_2026,
  author  = {Freitas, Elton},
  title   = {econcomplex: economic complexity and regional science indicators in Python},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/eltonfreitas/econcomplex}
}
```

Cite também os artigos originais dos indicadores utilizados — lista completa
na documentação em PDF. Referências centrais: Hidalgo & Hausmann (2009,
*PNAS*); Hidalgo et al. (2007, *Science*); Tacchella et al. (2012, *Sci.
Rep.*); Alshamsi, Pinheiro & Hidalgo (2018, *Nat. Commun.*); Pinheiro et al.
(2022, *Res. Policy*); Stojkoski & Hidalgo (2026, *Res. Policy*).

## Licença

MIT — veja [LICENSE](LICENSE).
