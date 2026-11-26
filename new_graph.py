from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_CSV = BASE_DIR / "dados" / "data.csv"
ARQUIVO_SAIDA = BASE_DIR / "resultados" / "q4_q7_desenvolvimento_uso_iag.png"

ORDEM_Q4 = [
    "Diariamente.",
    "Algumas vezes por semana.",
    "Algumas vezes por mês.",
    "Raramente.",
    "Nunca.",
]
ORDEM_Q7 = ["Diariamente.", "Frequentemente.", "Às vezes.", "Raramente."]
CORES = ["#1f4e78", "#4472c4", "#8faadc", "#d9e1f2"]


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


def gerar_grafico(caminho_csv=ARQUIVO_CSV, caminho_saida=ARQUIVO_SAIDA):
    caminho_csv = Path(caminho_csv)
    caminho_saida = Path(caminho_saida)
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    col_q4 = localizar_coluna(df, 4)
    col_q7 = localizar_coluna(df, 7)
    dados = df[[col_q4, col_q7]].dropna()
    if dados.empty:
        raise ValueError("Não há respostas válidas para o cruzamento entre Q4 e Q7.")

    ordem_q4 = ordenar_respostas(dados[col_q4], ORDEM_Q4)
    ordem_q7 = ordenar_respostas(dados[col_q7], ORDEM_Q7)
    tabela = pd.crosstab(dados[col_q4], dados[col_q7]).reindex(
        index=ordem_q4,
        columns=ordem_q7,
        fill_value=0,
    )
    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    rotulos_q4 = [
        f"{resposta.removesuffix('.')}\n(n={tabela.loc[resposta].sum()})"
        for resposta in ordem_q4
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
    esquerda = np.zeros(len(ordem_q4))
    posicoes_y = np.arange(len(ordem_q4))

    for i, resposta in enumerate(ordem_q7):
        valores = percentuais[resposta].to_numpy()
        ax.barh(
            posicoes_y,
            valores,
            left=esquerda,
            height=0.55,
            label=resposta.removesuffix("."),
            color=CORES[i % len(CORES)],
            edgecolor="white",
            linewidth=0.8,
        )
        for j, valor in enumerate(valores):
            if valor >= 10:
                cor_texto = "white" if i < 2 else "#1a1a1a"
                ax.text(
                    esquerda[j] + valor / 2,
                    posicoes_y[j],
                    f"{valor:.1f}%",
                    va="center",
                    ha="center",
                    color=cor_texto,
                    fontsize=9,
                    fontweight="bold",
                )
        esquerda += valores

    ax.set_yticks(posicoes_y)
    ax.set_yticklabels(rotulos_q4, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_ylabel("Frequência de desenvolvimento", fontsize=10)
    ax.set_xlabel("Proporção de participantes (%)", fontsize=10)
    ax.set_title(
        "Frequência de uso de IAG segundo a frequência de desenvolvimento",
        fontsize=11,
        fontweight="bold",
        pad=45,
    )
    ax.legend(
        title="Frequência de uso de IAG (Questão 7)",
        bbox_to_anchor=(0.5, 1.12),
        loc="center",
        ncol=len(ordem_q7),
        frameon=False,
        fontsize=8.8,
        title_fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")


if __name__ == "__main__":
    gerar_grafico()
