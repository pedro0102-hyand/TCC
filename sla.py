from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_CSV = BASE_DIR / "dados" / "data.csv"
ARQUIVO_SAIDA = BASE_DIR / "resultados" / "q15_resolucao_autonoma_perfil.png"

GRUPOS_Q15 = [
    "Frequentemente / Sempre",
    "Às vezes",
    "Raramente / Nunca",
]
MAPA_Q15 = {
    "Frequentemente.": GRUPOS_Q15[0],
    "Sempre.": GRUPOS_Q15[0],
    "Às vezes.": GRUPOS_Q15[1],
    "Raramente.": GRUPOS_Q15[2],
    "Nunca.": GRUPOS_Q15[2],
}
ORDEM_Q3 = ["Iniciante.", "Básico.", "Intermediário.", "Avançado."]
ORDEM_Q5 = [
    "Não tenho experiência.",
    "Menos de 1 ano.",
    "De 1 a 4 anos.",
    "Mais de 4 anos.",
]
ROTULOS_Q5 = {
    "Não tenho experiência.": "Sem experiência",
    "Menos de 1 ano.": "Menos de 1 ano",
    "De 1 a 4 anos.": "De 1 a 4 anos",
    "Mais de 4 anos.": "Mais de 4 anos",
}
CORES = ["#1f4e78", "#8faadc", "#d9e1f2"]


def localizar_coluna(df, numero):
    prefixo = re.compile(rf"^\s*{numero}\s*[\).\-:]")
    for coluna in df.columns:
        if prefixo.match(str(coluna)):
            return coluna

    raise ValueError(f"Não foi possível localizar a coluna da questão {numero}.")


def ordenar_respostas(respostas, ordem):
    presentes = list(respostas.dropna().unique())
    ordenadas = [resposta for resposta in ordem if resposta in presentes]
    return ordenadas + [resposta for resposta in presentes if resposta not in ordenadas]


def criar_tabela(dados, coluna_perfil, ordem):
    categorias = ordenar_respostas(dados[coluna_perfil], ordem)
    tabela = pd.crosstab(dados[coluna_perfil], dados["Q15_grupo"]).reindex(
        index=categorias,
        columns=GRUPOS_Q15,
        fill_value=0,
    )
    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    return tabela, percentuais


def plot_painel(ax, percentuais, rotulos, titulo):
    esquerda = np.zeros(len(percentuais))
    posicoes_y = np.arange(len(percentuais))

    for i, grupo in enumerate(GRUPOS_Q15):
        valores = percentuais[grupo].to_numpy()
        ax.barh(
            posicoes_y,
            valores,
            left=esquerda,
            height=0.55,
            label=grupo,
            color=CORES[i],
            edgecolor="white",
            linewidth=0.8,
        )
        for j, valor in enumerate(valores):
            if valor >= 10:
                cor_texto = "white" if i == 0 else "#1a1a1a"
                ax.text(
                    esquerda[j] + valor / 2,
                    posicoes_y[j],
                    f"{valor:.1f}%",
                    va="center",
                    ha="center",
                    color=cor_texto,
                    fontsize=8.8,
                    fontweight="bold",
                )
        esquerda += valores

    ax.set_yticks(posicoes_y)
    ax.set_yticklabels(rotulos, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proporção de participantes (%)", fontsize=9.5)
    ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def gerar_grafico(caminho_csv=ARQUIVO_CSV, caminho_saida=ARQUIVO_SAIDA):
    caminho_csv = Path(caminho_csv)
    caminho_saida = Path(caminho_saida)
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    col_q3 = localizar_coluna(df, 3)
    col_q5 = localizar_coluna(df, 5)
    col_q15 = localizar_coluna(df, 15)

    dados = df[[col_q3, col_q5, col_q15]].copy()
    for coluna in dados.columns:
        dados[coluna] = dados[coluna].astype("string").str.strip()
    dados = dados.dropna(subset=[col_q15])
    if dados.empty:
        raise ValueError("Não há respostas válidas para o cruzamento entre Q3, Q5 e Q15.")

    respostas_q15 = set(dados[col_q15].unique())
    desconhecidas = respostas_q15 - MAPA_Q15.keys()
    if desconhecidas:
        raise ValueError(
            "Há respostas da Q15 sem classificação: "
            + ", ".join(sorted(desconhecidas))
        )
    dados["Q15_grupo"] = dados[col_q15].map(MAPA_Q15)

    tabela_q3, percentuais_q3 = criar_tabela(dados, col_q3, ORDEM_Q3)
    tabela_q5, percentuais_q5 = criar_tabela(dados, col_q5, ORDEM_Q5)
    rotulos_q3 = [
        f"{categoria.removesuffix('.')}\n(n={tabela_q3.loc[categoria].sum()})"
        for categoria in tabela_q3.index
    ]
    rotulos_q5 = [
        f"{ROTULOS_Q5.get(categoria, categoria.removesuffix('.'))}\n"
        f"(n={tabela_q5.loc[categoria].sum()})"
        for categoria in tabela_q5.index
    ]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.3),
        dpi=300,
        sharey=False,
    )
    plot_painel(
        ax1,
        percentuais_q3,
        rotulos_q3,
        "(a) Por nível de conhecimento (Questão 3)",
    )
    plot_painel(
        ax2,
        percentuais_q5,
        rotulos_q5,
        "(b) Por tempo de experiência (Questão 5)",
    )

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Frequência de tentativa prévia sem IA (Questão 15)",
        bbox_to_anchor=(0.5, 1.08),
        loc="center",
        ncol=len(GRUPOS_Q15),
        frameon=False,
        fontsize=9.2,
        title_fontsize=9.5,
    )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")


if __name__ == "__main__":
    gerar_grafico()
