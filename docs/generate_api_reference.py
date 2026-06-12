"""
Gera as tabelas de referência da API (api_reference_en.tex e
api_reference_pt.tex) por introspecção do pacote instalado.

Uso:  python docs/generate_api_reference.py
Depois recompile os PDFs (tectonic docs/econcomplex_documentation_*.tex).
"""

import inspect
import os
import re

import econcomplex as ec

HERE = os.path.dirname(os.path.abspath(__file__))

GROUP_ORDER = ["core", "complexity", "relatedness", "specialization",
               "inequality", "productivity", "patents", "dynamics",
               "outlook", "optimization", "pipeline"]

GROUP_TITLES = {
    "core": ("Core", "Core"),
    "complexity": ("Complexity", "Complexidade"),
    "relatedness": ("Relatedness and Product Space", "Relatedness e Product Space"),
    "specialization": ("Regional Specialization", "Especializa\\c{c}\\~{a}o Regional"),
    "inequality": ("Inequality and Concentration", "Desigualdade e Concentra\\c{c}\\~{a}o"),
    "productivity": ("Productivity", "Produtividade"),
    "patents": ("Patent-Based Complexity", "Complexidade de Patentes"),
    "dynamics": ("Temporal Dynamics", "Din\\^{a}mica Temporal"),
    "outlook": ("Complexity Outlook", "Complexity Outlook"),
    "optimization": ("ECI Optimization and Strategic Diffusion",
                     "Otimiza\\c{c}\\~{a}o de ECI e Difus\\~{a}o Estrat\\'{e}gica"),
    "pipeline": ("Unified Pipeline", "Pipeline Unificado"),
}

# Descrições em português (uma linha por função canônica)
PT = {
    "rca": "Vantagem Comparativa Revelada (\\'{i}ndice de Balassa).",
    "rpop": "RCA normalizada por popula\\c{c}\\~{a}o (RPOP).",
    "mcp": "Matriz bin\\'{a}ria de presen\\c{c}a Mcp (RCA, RPOP, ambas ou manual).",
    "diversity": "Diversidade: n\\'{u}mero de atividades com RCA acima do limiar, por regi\\~{a}o.",
    "ubiquity": "Ubiquidade: n\\'{u}mero de regi\\~{o}es com RCA acima do limiar, por atividade.",
    "normalized_ubiquity": "Ubiquidade normalizada pela participa\\c{c}\\~{a}o no valor total.",
    "pivot_to_matrix": "Converte DataFrame longo em matriz larga (pivot).",
    "melt_matrix": "Converte matriz larga em DataFrame longo.",
    "binarize": "Binariza a matriz no limiar dado.",
    "normalize_zscore": "Padroniza um vetor (z-score).",
    "normalize_01": "Normaliza um vetor para [0, 1] (min--max).",
    "make_sample_data": "Gera dados sint\\'{e}ticos longos com estrutura aninhada (exemplos e testes).",
    "trim_core": "Poda iterativa ao n\\'{u}cleo (dmin, umin), removendo unidades degeneradas.",
    "eci_pci": "Porta de entrada \\'{u}nica do ECI/PCI (autovetor, reflex\\~{o}es ou fitness), com poda autom\\'{a}tica.",
    "eci_pci_eigenvector": "Implementa\\c{c}\\~{a}o do m\\'{e}todo do autovetor (Hidalgo \\& Hausmann 2009).",
    "method_of_reflections": "M\\'{e}todo das Reflex\\~{o}es iterativo para ECI/PCI.",
    "mor_regions": "Reflex\\~{o}es apenas do lado das regi\\~{o}es, no passo escolhido.",
    "mor_activities": "Reflex\\~{o}es apenas do lado das atividades.",
    "fitness_complexity": "Algoritmo Fitness--Complexity (Tacchella et al.\\ 2012).",
    "subnational_eci": "ECI subnacional projetado com PCI externo.",
    "proximity": "Matrizes de proximidade entre produtos e/ou regi\\~{o}es (discreta ou cont\\'{i}nua).",
    "continuous_proximity": "Proximidade cont\\'{i}nua a partir do RCA (correla\\c{c}\\~{a}o ou cosseno).",
    "cosine_proximity": "Atalho: proximidade cont\\'{i}nua por cosseno.",
    "correlation_proximity": "Atalho: proximidade cont\\'{i}nua por correla\\c{c}\\~{a}o.",
    "relatedness_density": "Densidade de relatedness por par (regi\\~{a}o, atividade), em percentual.",
    "distance": "Dist\\^{a}ncia: 1 -- densidade/100.",
    "relatedness_density_internal": "Densidade restrita \\`{a}s atividades j\\'{a} presentes.",
    "relatedness_density_external": "Densidade restrita \\`{a}s atividades ausentes.",
    "relative_relatedness": "Relatedness relativa: z-score no option set (Pinheiro et al.\\ 2022).",
    "co_occurrence": "Matriz de coocorr\\^{e}ncia entre atividades.",
    "relatedness_index": "\\'{I}ndice de relatedness por coocorr\\^{e}ncia (probabilidade, associa\\c{c}\\~{a}o, cosseno, Jaccard).",
    "z_score_novelty": "Z-score de novidade (atipicidade da coocorr\\^{e}ncia).",
    "cross_proximity": "Proximidade entre dois espa\\c{c}os de atividades distintos.",
    "cross_relatedness": "Densidade de relatedness cross-space (regi\\~{o}es x espa\\c{c}o B).",
    "location_quotient": "Quociente Locacional (id\\^{e}ntico ao RCA).",
    "location_quotient_avg": "LQ m\\'{e}dio ponderado por regi\\~{a}o (coeficiente de Hoover).",
    "hachman_index": "\\'{I}ndice de Hachman (similaridade \\`{a} estrutura nacional).",
    "specialization_coefficient": "Coeficiente de especializa\\c{c}\\~{a}o de Hoover.",
    "krugman_index": "\\'{I}ndice de especializa\\c{c}\\~{a}o de Krugman.",
    "export_similarity": "Similaridade de pauta entre regi\\~{o}es (Bahar et al.\\ 2014).",
    "gini": "Coeficiente de Gini padr\\~{a}o (vetor ou colunas de DataFrame).",
    "locational_gini": "Gini locacional de Krugman, por atividade.",
    "hoover_gini": "Gini com eixo populacional (curva de Hoover).",
    "herfindahl": "\\'{I}ndice Herfindahl--Hirschman por regi\\~{a}o.",
    "shannon_entropy": "Entropia de Shannon por regi\\~{a}o (diversifica\\c{c}\\~{a}o).",
    "hoover_index": "\\'{I}ndice de Hoover (Robin Hood) por atividade.",
    "prody": "PRODY: n\\'{i}vel de renda associado a cada atividade.",
    "expy": "EXPY: renda impl\\'{i}cita da cesta de cada regi\\~{a}o.",
    "product_gini_index": "PGI: desigualdade embutida em cada produto.",
    "product_emissions_index": "PEII: intensidade de emiss\\~{o}es embutida em cada produto.",
    "ease_of_recombination": "Facilidade de recombina\\c{c}\\~{a}o (EOR) por tecnologia.",
    "modular_complexity": "Complexidade modular de cada patente.",
    "modular_complexity_avg": "Complexidade modular m\\'{e}dia por tecnologia.",
    "growth_rate": "Taxa de crescimento agregada entre dois per\\'{i}odos.",
    "growth_matrix": "Matriz de crescimento c\\'{e}lula a c\\'{e}lula.",
    "growth_rates": "Crescimento em painel longo, entre per\\'{i}odos consecutivos.",
    "entry": "Entradas: transi\\c{c}\\~{o}es 0 para 1 na especializa\\c{c}\\~{a}o.",
    "exit": "Sa\\'{i}das: transi\\c{c}\\~{o}es 1 para 0 na especializa\\c{c}\\~{a}o.",
    "entry_exit_summary": "Resumo de entradas e sa\\'{i}das por par (regi\\~{a}o, atividade).",
    "entry_tracking": "Entradas em formato longo (painel).",
    "exit_tracking": "Sa\\'{i}das em formato longo (painel).",
    "complexity_outlook_index": "COI: potencial de diversifica\\c{c}\\~{a}o rumo a atividades complexas.",
    "complexity_outlook_gain": "COG: ganho de COI ao desenvolver cada atividade.",
    "calibrate_steppingstone": "Calibra o modelo forward com steppingstone (OLS de entrada/sa\\'{i}da).",
    "effort_matrix": "Esfor\\c{c}o $W_{cp}$: RCA adicional necess\\'{a}rio para entrar em cada atividade.",
    "forecast_specialization": "Proje\\c{c}\\~{a}o sem pol\\'{i}tica (W=0): RCA, Mcp, PCI e ECI futuros.",
    "eci_optimization": "Programa 0--1: portf\\'{o}lio de menor esfor\\c{c}o para atingir o ECI alvo.",
    "calibrate_growth_model": "Regress\\~{a}o de crescimento em painel (ECI, converg\\^{e}ncia, intera\\c{c}\\~{a}o).",
    "expected_growth": "Crescimento anual esperado dado o ECI e o PIB per capita.",
    "eci_target_for_growth": "Inverte a regress\\~{a}o: ECI compat\\'{i}vel com a meta de crescimento.",
    "proximity_network": "Rede bin\\'{a}ria de atividades relacionadas (phi acima do limiar).",
    "activation_probabilities": "Probabilidade de ativa\\c{c}\\~{a}o $p = B x^{\\alpha}$ por atividade.",
    "calibrate_contagion": "Calibra B e alfa do cont\\'{a}gio a partir das entradas observadas.",
    "diversification_strategy": "Sequ\\^{e}ncia de alvos por estrat\\'{e}gia heur\\'{i}stica, com tempos esperados.",
    "expected_diversification_time": "Tempo total esperado de uma sequ\\^{e}ncia fixa de alvos.",
    "compare_strategies": "Compara o tempo total das cinco estrat\\'{e}gias heur\\'{i}sticas.",
    "optimize_sequence": "Recozimento simulado sobre sequ\\^{e}ncias; nunca pior que a gulosa.",
    "compute_complexity": "Pipeline completo: todos os indicadores a partir do DataFrame longo (suporta painel).",
}


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            .replace("^", r"\^{}").replace("~", r"\~{}")
            .replace("→", "->").replace("—", "--"))


def short_signature(fn):
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return "(...)"
    parts, kw_marker = [], False
    for p in sig.parameters.values():
        if p.kind == p.KEYWORD_ONLY and not kw_marker:
            parts.append("*")
            kw_marker = True
        if p.default is inspect.Parameter.empty:
            parts.append(p.name)
        else:
            d = p.default
            rep = f"'{d}'" if isinstance(d, str) else repr(d)
            parts.append(f"{p.name}={rep}")
    return "(" + ", ".join(parts) + ")"


def en_description(fn, alias_of):
    if alias_of:
        return f"Alias of \\texttt{{{esc(alias_of)}}}."
    doc = inspect.getdoc(fn) or ""
    first = doc.split("\n\n")[0].replace("\n", " ")
    first = re.sub(r"\s+", " ", first).strip()
    m = re.match(r"(.+?\.)\s", first + " ")
    return esc(m.group(1) if m else first)


def pt_description(fn, name, alias_of):
    if alias_of:
        return f"Alias de \\texttt{{{esc(alias_of)}}}."
    return PT.get(name, en_description(fn, None))


def build(lang):
    groups = {}
    for name in ec.__all__:
        fn = getattr(ec, name)
        if not callable(fn):
            continue
        mod = fn.__module__.replace("econcomplex.", "").split(".")[0]
        alias_of = fn.__name__ if fn.__name__ != name else ""
        desc = (en_description(fn, alias_of) if lang == "en"
                else pt_description(fn, name, alias_of))
        sig = esc(short_signature(fn))
        groups.setdefault(mod, []).append((esc(name), sig, desc))

    head_fn = "Function" if lang == "en" else "Fun\\c{c}\\~{a}o"
    head_ds = "Description" if lang == "en" else "Descri\\c{c}\\~{a}o"
    out = []
    for g in GROUP_ORDER:
        if g not in groups:
            continue
        title = GROUP_TITLES[g][0 if lang == "en" else 1]
        out.append(f"\\subsection{{\\texttt{{{g}}} --- {title}}}")
        out.append(
            "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{9.2cm} "
            ">{\\raggedright\\arraybackslash}p{7.0cm}}")
        out.append("\\toprule")
        out.append(f"\\textbf{{{head_fn}}} & \\textbf{{{head_ds}}} \\\\")
        out.append("\\midrule\\endhead")
        for name, sig, desc in groups[g]:
            out.append(f"\\texttt{{\\footnotesize {name}{sig}}} & "
                       f"{{\\small {desc}}} \\\\[2pt]")
        out.append("\\bottomrule")
        out.append("\\end{longtable}")
        out.append("")
    path = os.path.join(HERE, f"api_reference_{lang}.tex")
    with open(path, "w") as f:
        f.write("\n".join(out))
    n = sum(len(v) for v in groups.values())
    print(f"{path}: {n} funções")


if __name__ == "__main__":
    build("en")
    build("pt")
