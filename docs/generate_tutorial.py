"""
Gera o tutorial passo a passo da documentação.

Cada passo abaixo tem o código, o texto que o antecede e o texto que
explica a saída, em inglês e em português. O script executa o código,
captura o que ele imprime e escreve:

    docs/tutorial_en.tex, docs/tutorial_pt.tex   (incluídos nos PDFs)
    examples/tutorial.py                          (o mesmo código, corrido)

Assim os números do tutorial são sempre os que a biblioteca produz. No
texto, <<expressão>> é substituído pelo valor da expressão depois do
passo (<<expressão|1>> fixa uma casa decimal). Cada passo pode ter um
`check`: se a saída deixar de sustentar o que o texto afirma, a geração
falha em vez de publicar um texto errado.

Uso:  PYTHONPATH=. python docs/generate_tutorial.py
Precisa do scikit-learn (passo 9). Depois recompile os PDFs.
"""

import contextlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH = 70                      # largura útil dos blocos de código no PDF

SETUP = '''import warnings

import numpy as np
import pandas as pd
import econcomplex as ec

warnings.simplefilter("ignore")
pd.set_option("display.width", 68)
'''

INTRO = (
    r"""This tutorial follows one analysis from the raw table to a shortlist
of diversification targets for a region, and then tests whether the
method behind the shortlist predicts where regions actually went. Each
step shows the code, what it prints and how to read it. The data are
synthetic (\fn{make\_sample\_data}), so every number below is reproduced
by running \texttt{examples/tutorial.py}; with your own data, replace
Step 1 and keep the rest.

The example follows region \texttt{L045}, which has an average number of
specializations. Region and activity labels are arbitrary codes.""",
    r"""Este tutorial acompanha uma análise do começo ao fim: da tabela
bruta até uma lista curta de alvos de diversificação para uma região, e
depois o teste de se o método que gerou a lista prevê para onde as
regiões de fato foram. Cada passo mostra o código, o que ele imprime e
como ler o resultado. Os dados são sintéticos
(\fn{make\_sample\_data}), de modo que todo número abaixo é reproduzido
ao rodar \texttt{examples/tutorial.py}; com os seus dados, basta trocar o
Passo 1 e manter o resto.

O exemplo acompanha a região \texttt{L045}, que tem um número mediano de
especializações. Os rótulos de regiões e atividades são códigos
arbitrários.""",
)

STEPS = [
    dict(
        title=("From the Table to the Matrix", "Da Tabela à Matriz"),
        before=(
            r"""The library starts from a long table: one row per region and
activity, with a value (employment, exports, patents).
\fn{pivot\_to\_matrix} turns it into the regions $\times$ activities
matrix that every function takes.""",
            r"""A biblioteca parte de uma tabela longa: uma linha por região e
atividade, com um valor (emprego, exportações, patentes).
\fn{pivot\_to\_matrix} a transforma na matriz regiões $\times$
atividades que todas as funções recebem.""",
        ),
        code='''df = ec.make_sample_data(n_locs=60, n_acts=40, seed=7)
print(df.head(3))
print(len(df), "rows |", df["loc"].nunique(), "regions |",
      df["act"].nunique(), "activities")

mat = ec.pivot_to_matrix(df, "loc", "act", "val")
print("matrix:", mat.shape)
''',
        after=(
            r"""With your own file the first line becomes
\texttt{df = pd.read\_csv("my\_data.csv")} and the three column names
change. Two things are worth checking before going on, because the
pivot hides them: pairs of region and activity that appear twice are
summed, and pairs that are absent become zero. Negative or missing
values are rejected by the functions with an error that names the
problem.""",
            r"""Com o seu arquivo, a primeira linha vira
\texttt{df = pd.read\_csv("meus\_dados.csv")} e mudam os três nomes de
coluna. Vale conferir duas coisas antes de seguir, porque o pivot as
esconde: pares de região e atividade repetidos são somados, e pares
ausentes viram zero. Valores negativos ou faltantes são recusados pelas
funções, com um erro que nomeia o problema.""",
        ),
    ),
    dict(
        title=("Who Is Specialized in What", "Quem É Especializado em Quê"),
        before=(
            r"""Raw values mostly reflect size: a large region has more of
everything. The RCA (the location quotient, when the data are
employment) divides the share of the activity in the region by its share
in the whole economy, and a region counts as specialized where
$\text{RCA} \geq 1$. The binary matrix $M$ records those
specializations; its row sums are the diversity of each region and its
column sums the ubiquity of each activity.""",
            r"""Os valores brutos refletem sobretudo o tamanho: uma região grande
tem mais de tudo. A VCR (o quociente locacional, quando o dado é
emprego) divide a participação da atividade na região pela participação
dela na economia inteira, e a região conta como especializada onde
$\text{VCR} \geq 1$. A matriz binária $M$ registra essas
especializações; as somas das linhas são a diversidade de cada região e
as das colunas, a ubiquidade de cada atividade.""",
        ),
        code='''rca = ec.rca(mat)
M = ec.mcp(mat)                    # 1 where RCA >= 1
diversity = ec.diversity(mat)      # specializations per region
ubiquity = ec.ubiquity(mat)        # specialized regions per activity

region = "L045"
print(rca.loc[region].nlargest(3).round(2))
print("specializations of", region, "=", int(diversity[region]))
print("median diversity =", diversity.median(),
      "| median ubiquity =", ubiquity.median())
''',
        after=(
            r"""\texttt{L045} is specialized in <<int(diversity[region])>> of the
40 activities, and the median region in <<int(diversity.median())>>. Its
strongest specialization has an RCA of <<rca.loc[region].max()>>: the
activity weighs that many times more in the region than in the economy
as a whole. The threshold of 1 is the \texttt{threshold} argument of
every function that binarizes, and \texttt{use\_rca=False} skips the RCA
when the matrix is already an index or a 0/1 table.""",
            r"""A \texttt{L045} é especializada em <<int(diversity[region])>> das 40
atividades, e a região mediana em <<int(diversity.median())>>. A sua
especialização mais forte tem VCR de <<rca.loc[region].max()>>: a
atividade pesa esse número de vezes mais na região do que na economia
como um todo. O limiar de 1 é o argumento \texttt{threshold} de toda
função que binariza, e \texttt{use\_rca=False} dispensa a VCR quando a
matriz já é um índice ou uma tabela 0/1.""",
        ),
    ),
    dict(
        title=("Complexity, and How Far to Trust It",
               "Complexidade, e Quanto Confiar Nela"),
        before=(
            r"""The ECI ranks regions by the complexity of what they are
specialized in, and the PCI ranks activities by the complexity of the
regions that have them. \fn{eci\_pci} returns the two; the default is
the eigenvector method, and \texttt{method=} switches to reflections,
Fitness or GENEPY in the same call. \fn{eci\_diagnostics} then says how
firm the ranking is for this particular matrix.""",
            r"""O ECI ordena as regiões pela complexidade daquilo em que são
especializadas, e o PCI ordena as atividades pela complexidade das
regiões que as têm. \fn{eci\_pci} devolve os dois; o padrão é o método
do autovetor, e \texttt{method=} troca para reflexos, Fitness ou GENEPY
na mesma chamada. \fn{eci\_diagnostics} diz em seguida quão firme é o
ranking nesta matriz em particular.""",
        ),
        code='''eci, pci = ec.eci_pci(mat)
order = eci.sort_values(ascending=False)
print(order.round(2).iloc[[0, 1, -2, -1]])
print("ECI of", region, "=", round(eci[region], 2), "| rank",
      int(eci.rank(ascending=False)[region]), "of", len(eci))

diag = ec.eci_diagnostics(mat, n_boot=100, seed=1)
keys = ["gap", "corr_diversity", "rank_stability_mean", "sign_flips"]
print(diag[keys].round(3))
''',
        check='assert diag["corr_diversity"] > 0.5 and diag["sign_flips"] == 0',
        after=(
            r"""Both indices are z-scores of this matrix: \texttt{L045} has an ECI
of <<eci[region]>>, rank <<int(eci.rank(ascending=False)[region])>> of
60. Because they are relative, the ECI of two years or of two
classifications cannot be compared in level, only in rank; a region or
activity returned as \texttt{NaN} was left out for having no
specialization at all. The diagnostics are reassuring here: the ECI has
a correlation of <<diag["corr_diversity"]>> with diversity, and when a
fifth of the activities is dropped at random the ranking keeps a rank
correlation of <<diag["rank_stability_mean"]>> with the original and
never comes out inverted. In small or sparse matrices these numbers are
often much worse. If the correlation with diversity is close to zero,
the sign of the index is arbitrary, and it should be checked against
income or wages before ``high ECI'' is read as ``complex''.""",
            r"""Os dois índices são escores z desta matriz: a \texttt{L045} tem ECI
de <<eci[region]>>, posição <<int(eci.rank(ascending=False)[region])>>
entre 60. Por serem relativos, o ECI de dois anos ou de duas
classificações não se compara em nível, só em posição; uma região ou
atividade devolvida como \texttt{NaN} ficou de fora por não ter nenhuma
especialização. Os diagnósticos aqui tranquilizam: o ECI tem correlação
de <<diag["corr_diversity"]>> com a diversidade e, quando um quinto das
atividades é retirado ao acaso, o ranking mantém correlação de postos de
<<diag["rank_stability_mean"]>> com o original e nunca sai invertido. Em
matrizes pequenas ou esparsas esses números costumam ser bem piores. Se
a correlação com a diversidade for próxima de zero, o sinal do índice é
arbitrário, e convém conferi-lo com renda ou salários antes de ler ``ECI
alto'' como ``complexo''.""",
        ),
    ),
    dict(
        title=("What Is Within Reach", "O Que Está ao Alcance"),
        before=(
            r"""Two activities are close when the same regions tend to be
specialized in both. From the proximity matrix $\phi$, relatedness
density measures, for each region and activity, how much of what is
related to the activity the region already has. It is the library's
basic measure of how feasible an activity is for a region.
\fn{relatedness\_density\_external} keeps only the activities the region
does not have, which are its options.""",
            r"""Duas atividades são próximas quando as mesmas regiões tendem a ser
especializadas nas duas. A partir da matriz de proximidade $\phi$, a
densidade de relatedness mede, para cada região e atividade, quanto do
que é relacionado à atividade a região já tem. É a medida básica da
biblioteca para a viabilidade de uma atividade numa região.
\fn{relatedness\_density\_external} guarda só as atividades que a
região não tem, que são as suas opções.""",
        ),
        code='''phi = ec.product_proximity(mat)                  # proximity matrix
density = ec.relatedness_density(mat, phi=phi)   # 0 to 100
options = ec.relatedness_density_external(mat, phi=phi)

nearest = options.loc[region].nlargest(4)
print(nearest.round(1))

best = nearest.index[0]
why = ec.density_contributions(mat, region, best, phi=phi)
print(why.head(4).round(1))
print("sum of all contributions =", round(why.sum(), 1))
''',
        after=(
            r"""The closest activity \texttt{L045} lacks is \texttt{<<best>>}, with
a density of <<nearest.iloc[0]|1>>: weighting by proximity, the region
has that percentage of what is related to it. \fn{density\_contributions}
opens the number. It lists the activities of the region that bring
\texttt{<<best>>} closer, and their contributions add up to the density.
This is the sentence that goes into a regional profile:
``\texttt{<<best>>} is within reach because the region already has
\texttt{<<why.index[0]>>}, \texttt{<<why.index[1]>>} and
\texttt{<<why.index[2]>>}''.""",
            r"""A atividade mais próxima que a \texttt{L045} não tem é a
\texttt{<<best>>}, com densidade de <<nearest.iloc[0]|1>>: ponderando
pela proximidade, a região tem esse percentual do que é relacionado a
ela. \fn{density\_contributions} abre o número. Ela lista as atividades
da região que aproximam a \texttt{<<best>>}, e as contribuições somam a
densidade. É a frase que entra na ficha de uma região: ``a
\texttt{<<best>>} está ao alcance porque a região já tem
\texttt{<<why.index[0]>>}, \texttt{<<why.index[1]>>} e
\texttt{<<why.index[2]>>}''.""",
        ),
    ),
    dict(
        title=("What Is Worth Reaching", "O Que Vale a Pena Alcançar"),
        before=(
            r"""Feasibility says what is within reach, not what is worth
reaching. The second axis is a choice of the analyst: complexity, wages,
growth, emissions. \fn{desirability\_index} standardizes each indicator
and takes their weighted mean, with the weights written down. The wage
series below is simulated and stands for an external source.""",
            r"""A viabilidade diz o que está ao alcance, não o que vale a pena
alcançar. O segundo eixo é uma escolha de quem analisa: complexidade,
salários, crescimento, emissões. \fn{desirability\_index} padroniza cada
indicador e tira a média ponderada, com os pesos declarados. A série de
salários abaixo é simulada e faz as vezes de uma fonte externa.""",
        ),
        code='''rng = np.random.default_rng(7)
wage = 1000 * np.exp(0.3 * pci + rng.normal(0, 0.3, len(pci)))
indicators = pd.DataFrame({"pci": pci, "wage": wage})
weights = {"pci": 0.6, "wage": 0.4}

desirability = ec.desirability_index(indicators, weights)
robust = ec.weight_sensitivity(indicators, weights, top=10, seed=1)
print(robust.head(4).round(2))
''',
        after=(
            r"""The weights are a judgment, so the ranking should not hang on
them. \fn{weight\_sensitivity} redraws the weights a thousand times
around the chosen ones and records where each activity lands.
\texttt{<<robust.index[0]>>} is first with the chosen weights and stays
among the ten best in <<100 * robust["top_share"].iloc[0]|0>>\,\% of the
draws. When \texttt{rank\_p05} and \texttt{rank\_p95} are far apart, the
position of the activity depends on the weights, and the report should
say so. An indicator that counts against an activity, such as emissions,
takes a negative weight.""",
            r"""Os pesos são um julgamento, e o ranking não deveria depender
deles. \fn{weight\_sensitivity} sorteia os pesos mil vezes em torno dos
escolhidos e registra onde cada atividade vai parar. A
\texttt{<<robust.index[0]>>} é a primeira com os pesos escolhidos e fica
entre as dez melhores em <<100 * robust["top_share"].iloc[0]|0>>\,\% dos
sorteios. Quando \texttt{rank\_p05} e \texttt{rank\_p95} ficam distantes,
a posição da atividade depende dos pesos, e o relatório deve dizer isso.
Um indicador que conta contra a atividade, como emissões, entra com peso
negativo.""",
        ),
    ),
    dict(
        title=("The Shortlist", "A Lista Curta"),
        before=(
            r"""With the two axes, each option of the region falls in one of
four quadrants. The cuts are the medians of the region's own options, so
``feasible'' means feasible compared with what else this region could
do. \fn{opportunity\_alignment} summarizes the picture in one number:
the correlation between the two axes among the region's options.""",
            r"""Com os dois eixos, cada opção da região cai em um de quatro
quadrantes. Os cortes são as medianas das opções da própria região, de
modo que ``viável'' quer dizer viável em comparação com o que mais essa
região poderia fazer. \fn{opportunity\_alignment} resume o quadro num
número: a correlação entre os dois eixos nas opções da região.""",
        ),
        code='''quadrants = ec.strategy_quadrants(options, desirability)
print(quadrants.loc[region].value_counts())

alignment = ec.opportunity_alignment(options, desirability)
print("alignment of", region, "=", round(alignment[region], 2))

shortlist = pd.DataFrame({
    "density": options.loc[region],
    "desirability": desirability,
    "quadrant": quadrants.loc[region],
}).dropna()
desirable = shortlist["quadrant"].isin(["short jump", "long bet"])
wanted = shortlist[desirable]
print(wanted.sort_values("density", ascending=False).head(4).round(2))
''',
        check=('n_jumps = int((shortlist["quadrant"] == "short jump").sum())\n'
               'ranked = wanted.sort_values("density", ascending=False)\n'
               'assert alignment[region] < -0.5 and n_jumps == 1 and len(wanted) >= 4\n'
               'assert list(ranked["quadrant"].iloc[:3]) == ["short jump", "long bet", "long bet"]'),
        after=(
            r"""Of the <<len(shortlist)>> activities \texttt{L045} lacks, only
<<n_jumps>> is a short jump (feasible and desirable);
<<int((shortlist["quadrant"] == "long bet").sum())>> are long bets
(desirable but far) and
<<int((shortlist["quadrant"] == "low value").sum())>> are within reach
but of low value. The alignment of <<alignment[region]>> says the same:
among this region's options, the closer an activity is, the less
desirable it tends to be. This is the usual situation of regions of low
or middling complexity, and it changes what the shortlist has to be.
With so few short jumps, the useful list is the desirable half of the
options ordered by density: the one short jump,
\texttt{<<ranked.index[0]>>}, followed by the long bets that are least
far, \texttt{<<ranked.index[1]>>} and \texttt{<<ranked.index[2]>>},
with densities between <<ranked["density"].iloc[2]|1>> and
<<ranked["density"].iloc[0]|1>>. In a region with positive alignment the
short jumps would fill the list by themselves. How many unrelated bets
to make, and in which order, is what \fn{optimize\_sequence} answers,
and \fn{eci\_optimization} returns the set of new activities that
reaches a target ECI with the least effort (see the sections on
Strategic Diffusion and ECI Optimization).""",
            r"""Das <<len(shortlist)>> atividades que a \texttt{L045} não tem, só
<<n_jumps>> é salto curto (viável e desejável);
<<int((shortlist["quadrant"] == "long bet").sum())>> são apostas longas
(desejáveis, mas distantes) e
<<int((shortlist["quadrant"] == "low value").sum())>> estão ao alcance,
mas valem pouco. O alinhamento de <<alignment[region]>> diz o mesmo: nas
opções desta região, quanto mais próxima a atividade, menos desejável
ela tende a ser. É a situação comum de regiões de complexidade baixa ou
intermediária, e ela muda o que a lista curta precisa ser. Com tão
poucos saltos curtos, a lista útil é a metade desejável das opções
ordenada pela densidade: o único salto curto, a
\texttt{<<ranked.index[0]>>}, seguido das apostas longas menos
distantes, a \texttt{<<ranked.index[1]>>} e a
\texttt{<<ranked.index[2]>>}, com densidades entre
<<ranked["density"].iloc[2]|1>> e <<ranked["density"].iloc[0]|1>>. Numa
região
com alinhamento positivo, os saltos curtos encheriam a lista sozinhos.
Quantas apostas não relacionadas fazer, e em que ordem, é o que
\fn{optimize\_sequence} responde, e \fn{eci\_optimization} devolve o
conjunto de novas atividades que alcança um ECI-alvo com o menor esforço
(ver as seções de Difusão Estratégica e de Otimização de ECI).""",
        ),
    ),
    dict(
        title=("Reading the Result Through Policy Lists",
               "Lendo o Resultado por Listas de Política"),
        before=(
            r"""Policies are usually written as lists of activities: a mission,
a value chain, a green taxonomy. The list functions read the same matrix
through a list. Below, the last fifteen activities stand for a mission
and three groups of activities for the links of a chain.""",
            r"""Políticas costumam ser escritas como listas de atividades: uma
missão, uma cadeia produtiva, uma taxonomia verde. As funções de lista
leem a mesma matriz por meio de uma lista. Abaixo, as quinze últimas
atividades fazem as vezes de uma missão e três grupos de atividades, dos
elos de uma cadeia.""",
        ),
        code='''acts = list(mat.columns)
lists = {"mission": acts[25:]}
feasibility = density / 100            # shares, 0 to 1
tab = ec.list_complexity_potential(mat, pci, feasibility, lists)
print(tab.loc[region].round(2))

chain = {"inputs": acts[0:6], "components": acts[15:21],
         "assembly": acts[33:39]}
links = ec.missing_links(mat, chain, feasibility, cut=0.4)
print(links.loc[region])
''',
        after=(
            r"""\texttt{L045} is specialized in
<<int(tab.loc[(region, "mission"), "n_present"])>> of the
<<int(tab.loc[(region, "mission"), "n_list"])>> activities of the
mission, which here are the most complex ones. \texttt{presence} adds
the PCI of the activities it has from the list, and \texttt{potential}
is the mean PCI of those it lacks, each weighted by its density; with a list of green products these are the green
complexity index and potential. For the chain, a link is
\texttt{present} when the region is specialized in at least one of its
activities, \texttt{feasible} when some activity of the link has a
density of 40\,\% or more, and \texttt{missing} otherwise.
\fn{transition\_risk} applies the same logic to activities expected to
shrink.""",
            r"""A \texttt{L045} é especializada em
<<int(tab.loc[(region, "mission"), "n_present"])>> das
<<int(tab.loc[(region, "mission"), "n_list"])>> atividades da missão,
que aqui são as mais complexas. \texttt{presence} soma o PCI das
atividades da lista que ela tem, e \texttt{potential} é o PCI médio das
que faltam, cada uma ponderada pela sua densidade; com uma
lista de produtos verdes, são o índice e o potencial de complexidade
verde. Na cadeia, um elo é \texttt{present} quando a região é
especializada em pelo menos uma das suas atividades, \texttt{feasible}
quando alguma atividade do elo tem densidade de 40\,\% ou mais, e
\texttt{missing} caso contrário. \fn{transition\_risk} aplica a mesma
lógica a atividades que tendem a encolher.""",
        ),
    ),
    dict(
        title=("Does Density Predict Entry?", "A Densidade Prevê a Entrada?"),
        before=(
            r"""Everything so far assumes that density says where regions will
go. With two periods this can be tested: compute the scores with the
first period and see whether they rank the entries observed by the
second. Here the second period is simulated, with faster growth where
density is high; with real data, load the later year in its place.""",
            r"""Tudo até aqui supõe que a densidade diz para onde as regiões
vão. Com dois períodos isso pode ser testado: calculam-se os escores com
o primeiro período e verifica-se se eles ordenam as entradas observadas
até o segundo. Aqui o segundo período é simulado, com crescimento mais
rápido onde a densidade é alta; com dados reais, carregue o ano
posterior no lugar dele.""",
        ),
        code='''size = np.outer(mat.sum(axis=1), mat.sum(axis=0)) / mat.values.sum()
mat_t1 = (mat * rng.lognormal(0, 0.3, mat.shape)
          + size * rng.lognormal(0, 1.0, mat.shape)
          * 3 * (density / 100) ** 2)

entered, at_risk = ec.entry_targets(rca, ec.rca(mat_t1))
print(int(at_risk.values.sum()), "cells at risk |",
      int(entered.sum().sum()), "entries")

scores = {"density": density,
          "relative": ec.relative_relatedness(mat, phi=phi),
          "persistence": rca}
val = ec.validate_entry(scores, entered, k=(5,),
                        baseline="persistence", n_boot=200, seed=1)
cols = ["auc", "auc_within", "precision@5"]
print(val["value"].unstack()[cols].round(3))
print(val.loc[("density", "auc")].round(3))
''',
        check=('w = val["value"].unstack()["auc_within"]\n'
               'assert val.loc[("density", "auc"), "diff_lo"] > 0\n'
               'assert w["persistence"] < w["density"] < val.loc[("density", "auc"), "value"]\n'
               'assert w["density"] - w["persistence"] < val.loc[("density", "auc"), "diff"] / 2'),
        after=(
            r"""A cell is at risk when its RCA is below 0.5 at the start, and it
enters when the RCA reaches 1; the gap avoids counting cells that only
hover around 1. Of <<int(at_risk.values.sum())>> cells at risk,
<<int(entered.sum().sum())>> entered. The AUC is the probability that an
entry is ranked above a non-entry: <<val.loc[("density", "auc"), "value"]>>
for density and <<val.loc[("persistence", "auc"), "value"]>> for
persistence, the initial RCA. \texttt{auc\_within} computes the same
inside each region and averages. It is the number that matters for
choosing among the options of one region, and it is lower because part
of the pooled AUC comes from telling regions apart: inside regions,
density reaches <<val.loc[("density", "auc_within"), "value"]>> and
persistence <<val.loc[("persistence", "auc_within"), "value"]>>, a much
smaller distance.
\texttt{precision@5} is the share of each region's five best-ranked
options that entered. The last block compares density with persistence:
a gain of <<val.loc[("density", "auc"), "diff"]>> in AUC, with a 95\,\%
interval from <<val.loc[("density", "auc"), "diff_lo"]>> to
<<val.loc[("density", "auc"), "diff_hi"]>> obtained by resampling
regions. For the test to be honest, every score has to use data from the
first period only, the proximity matrix included.""",
            r"""Uma célula está em risco quando a VCR é menor que 0,5 no início,
e entra quando a VCR chega a 1; o intervalo evita contar células que só
oscilam em torno de 1. De <<int(at_risk.values.sum())>> células em risco,
<<int(entered.sum().sum())>> entraram. A AUC é a probabilidade de uma
entrada ficar acima de uma não entrada no ranking:
<<val.loc[("density", "auc"), "value"]>> para a densidade e
<<val.loc[("persistence", "auc"), "value"]>> para a persistência, que é
a VCR inicial. \texttt{auc\_within} calcula o mesmo dentro de cada região
e tira a média. É o número que importa para escolher entre as opções de
uma região, e é menor porque parte da AUC agregada vem de distinguir
regiões entre si: dentro das regiões, a densidade chega a
<<val.loc[("density", "auc_within"), "value"]>> e a persistência a
<<val.loc[("persistence", "auc_within"), "value"]>>, uma distância bem
menor. \texttt{precision@5} é a parcela das cinco opções mais
bem colocadas de cada região que entraram. O último bloco compara a
densidade com a persistência: ganho de
<<val.loc[("density", "auc"), "diff"]>> na AUC, com intervalo de 95\,\%
de <<val.loc[("density", "auc"), "diff_lo"]>> a
<<val.loc[("density", "auc"), "diff_hi"]>>, obtido reamostrando regiões.
Para o teste ser honesto, todo escore tem de usar só dados do primeiro
período, inclusive a matriz de proximidade.""",
        ),
    ),
    dict(
        title=("A Prediction Model and Expected Jobs (Optional)",
               "Um Modelo de Previsão e o Emprego Esperado (Opcional)"),
        before=(
            r"""With several measures of feasibility, a model can combine them.
\fn{build\_features} stacks the layers into one table of cells and adds
controls; \fn{predict\_entry} cross-validates by region, so that the
probability of each cell comes from a model that never saw its region.
This step needs scikit-learn
(\texttt{pip install "econcomplex[ml]"}).""",
            r"""Com várias medidas de viabilidade, um modelo pode combiná-las.
\fn{build\_features} empilha as camadas numa tabela de células e
acrescenta controles; \fn{predict\_entry} faz validação cruzada por
região, de modo que a probabilidade de cada célula vem de um modelo que
nunca viu a sua região. Este passo exige o scikit-learn
(\texttt{pip install "econcomplex[ml]"}).""",
        ),
        code='''layers = {"density": density, "implied": ec.implied_rca(mat)}
X = ec.build_features(mat, layers, at_risk=at_risk)
scores["model"] = ec.predict_entry(X, entered, model="logit", seed=1)
out = ec.validate_entry(scores, entered, k=(5,))["value"].unstack()
print(out.loc[["density", "model"], ["auc", "auc_within"]].round(3))

importance = ec.explain_entry(X, entered, model="logit", seed=1)
print(importance.head(3).round(3))

# next window: fit on t0 -> t1, predict from what is observed at t1
phi_t1 = ec.product_proximity(mat_t1)
layers_t1 = {"density": ec.relatedness_density(mat_t1, phi=phi_t1),
             "implied": ec.implied_rca(mat_t1)}
X_t1 = ec.build_features(mat_t1, layers_t1,
                         at_risk=ec.rca(mat_t1) < 0.5)
prob = ec.predict_entry(X, entered, model="logit",
                        new_features=X_t1, calibrate=True)
jobs = ec.potential_employment(prob, mat_t1)
print(jobs.groupby("region")["expected"].sum().nlargest(3).round(1))
''',
        check=('gap = (out.loc["model"] - out.loc["density"])[["auc", "auc_within"]]\n'
               'assert gap.abs().max() < 0.02 and importance.index[0] == "density"\n'
               'assert importance["importance"].iloc[1] < importance["importance"].iloc[0] / 4\n'
               'assert not importance["significant"].iloc[1:].any()'),
        after=(
            r"""The model enters \fn{validate\_entry} as one more score. Here it
ties with plain density (an AUC of <<out.loc["model", "auc"]|3>> against
<<out.loc["density", "auc"]|3>>, and
<<out.loc["model", "auc_within"]|3>> against
<<out.loc["density", "auc_within"]|3>> inside regions), which is what
one expects from data that were simulated from density alone. A model is worth its extra machinery
only when it wins this comparison out of sample. \fn{explain\_entry}
shuffles one feature at a time in the held-out regions and reports the
fall in AUC: density carries the prediction
(<<importance["importance"].iloc[0]>>) and no other feature is
statistically distinguishable from zero. For the next window the same function takes
\texttt{new\_features}: the model is fitted on the first window and
applied to what is observed at the end of it. \fn{potential\_employment}
then multiplies each calibrated probability by the size the activity
would have in the region with a location quotient of 1, and the sum by
region is the expected employment in new specializations.""",
            r"""O modelo entra em \fn{validate\_entry} como mais um escore. Aqui
ele empata com a densidade pura (AUC de <<out.loc["model", "auc"]|3>>
contra <<out.loc["density", "auc"]|3>>, e
<<out.loc["model", "auc_within"]|3>> contra
<<out.loc["density", "auc_within"]|3>> dentro das regiões), que é o que
se espera de dados simulados só a partir da densidade. Um modelo só compensa a maquinaria a mais
quando ganha essa comparação fora da amostra. \fn{explain\_entry}
embaralha um atributo de cada vez nas regiões fora do treino e informa a
queda da AUC: a densidade sustenta a previsão
(<<importance["importance"].iloc[0]>>) e nenhum outro atributo se
distingue estatisticamente de zero. Para a janela seguinte, a mesma função recebe
\texttt{new\_features}: o modelo é ajustado na primeira janela e
aplicado ao que se observa no fim dela. \fn{potential\_employment}
multiplica então cada probabilidade calibrada pelo tamanho que a
atividade teria na região com quociente locacional 1, e a soma por
região é o emprego esperado em novas especializações.""",
        ),
    ),
]

CHECKLIST = (
    r"""\subsection{Before Using the Results}

A few points account for most of the mistakes in applied work with these
indicators.

\begin{itemize}[leftmargin=1.5em]
  \item ECI and PCI are relative to one matrix. To follow a region over
    time, compare ranks, or project a fixed PCI on each year with
    \fn{subnational\_eci}.
  \item \texttt{NaN} means that the unit was left out (no
    specialization, or not an option); it is not a zero.
  \item In sparse data, where many regions have one or two
    specializations, compute complexity on the connected core with
    \texttt{dmin=2, umin=2}.
  \item Density comes in percent (0 to 100) by default and as a share
    with \texttt{percent=False}. Cuts such as the one of
    \fn{missing\_links} are read on the scale of the matrix you pass.
  \item External vectors (GDP, population, wages, PCI from another
    source) are matched to the matrix by label, in any order; a label
    that is missing raises an error instead of being filled.
  \item In any test of prediction, the proximity matrix and every score
    use the first period only.
  \item State the weights of the desirability index and report how the
    ranking moves when they change.
\end{itemize}
""",
    r"""\subsection{Antes de Usar os Resultados}

Alguns pontos respondem pela maior parte dos erros no uso aplicado
desses indicadores.

\begin{itemize}[leftmargin=1.5em]
  \item ECI e PCI são relativos a uma matriz. Para acompanhar uma região
    no tempo, compare posições, ou projete um PCI fixo em cada ano com
    \fn{subnational\_eci}.
  \item \texttt{NaN} quer dizer que a unidade ficou de fora (sem
    especialização, ou não é uma opção); não é um zero.
  \item Em dados esparsos, em que muitas regiões têm uma ou duas
    especializações, calcule a complexidade no núcleo conectado, com
    \texttt{dmin=2, umin=2}.
  \item A densidade vem em percentual (0 a 100) por padrão e como
    parcela com \texttt{percent=False}. Cortes como o de
    \fn{missing\_links} são lidos na escala da matriz que você passa.
  \item Vetores externos (PIB, população, salários, PCI de outra fonte)
    são casados com a matriz pelo rótulo, em qualquer ordem; um rótulo
    ausente gera erro, em vez de ser preenchido.
  \item Em qualquer teste de previsão, a matriz de proximidade e todos
    os escores usam só o primeiro período.
  \item Declare os pesos do índice de desejabilidade e informe como o
    ranking se move quando eles mudam.
\end{itemize}
""",
)

SECTION = ("Tutorial: From the Data to a List of Targets",
           "Tutorial: Dos Dados a uma Lista de Alvos")
STEP = ("Step", "Passo")
OUT_STYLE = (r"[language={}, numbers=none, backgroundcolor=\color{white}, "
             r"frame=leftline, framerule=2pt, rulecolor=\color{noteborder}, "
             r"xleftmargin=1.5em]")


def fill(text, ns, lang):
    """Replace <<expr>> / <<expr|decimals>> by the value of expr."""
    def value(match):
        expr, _, dec = match.group(1).rpartition("|")
        if not expr or not dec.strip().isdigit():
            expr, dec = match.group(1), "2"
        v = eval(" ".join(expr.split()), ns)
        if isinstance(v, float):
            v = f"{v:.{int(dec)}f}".replace("-", "$-$")
            if lang == 1:
                v = v.replace(".", "{,}")
        return str(v).replace("_", r"\_")
    return re.sub(r"<<(.+?)>>", value, text, flags=re.S)


def main():
    ns = {}
    exec(SETUP, ns)
    outputs = []
    for step in STEPS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(step["code"], ns)
        if "check" in step:
            exec(step["check"], ns)
        out = buf.getvalue().rstrip("\n")
        wide = [ln for ln in (step["code"] + out).splitlines() if len(ln) > WIDTH]
        if wide:
            raise SystemExit(f"linha com mais de {WIDTH} caracteres:\n" + "\n".join(wide))
        outputs.append(out)
        step["filled"] = [fill(step["after"][lang], ns, lang) for lang in (0, 1)]

    for lang, code in ((0, "en"), (1, "pt")):
        tex = [f"\\section{{{SECTION[lang]}}}", "", INTRO[lang], ""]
        for i, (step, out) in enumerate(zip(STEPS, outputs), start=1):
            tex += [f"\\subsection{{{STEP[lang]} {i}: {step['title'][lang]}}}", "",
                    step["before"][lang], "",
                    "\\begin{lstlisting}", step["code"].rstrip("\n"), "\\end{lstlisting}",
                    f"\\begin{{lstlisting}}{OUT_STYLE}", out, "\\end{lstlisting}", "",
                    step["filled"][lang], ""]
        tex += [CHECKLIST[lang]]
        with open(os.path.join(HERE, f"tutorial_{code}.tex"), "w") as f:
            f.write("\n".join(tex))

    script = ['"""', "Step-by-step tutorial of econcomplex",
              "====================================", "",
              "The code of the tutorial section of the PDF documentation, in one",
              "file. Generated by docs/generate_tutorial.py; edit the steps there.",
              "Run from the repository root:", "",
              "    python examples/tutorial.py", "",
              "Step 9 needs scikit-learn (pip install 'econcomplex[ml]').", '"""', "",
              SETUP]
    for i, step in enumerate(STEPS, start=1):
        script += ["", f"# ── Step {i}: {step['title'][0]} " + "─" * 20, step["code"].rstrip("\n")]
    with open(os.path.join(HERE, "..", "examples", "tutorial.py"), "w") as f:
        f.write("\n".join(script) + "\n")
    print(f"tutorial: {len(STEPS)} passos; tutorial_en.tex, tutorial_pt.tex, examples/tutorial.py")


if __name__ == "__main__":
    main()
