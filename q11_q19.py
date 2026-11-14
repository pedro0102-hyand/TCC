from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_CSV = BASE_DIR / "dados" / "data.csv"
ARQUIVO_SAIDA = BASE_DIR / "resultados" / "q11_q19_revisao_autonomia.png"

GRUPOS_Q11 = ["A", "B"]
MAPA_Q11 = {
    "Nunca.": "A",
    "Raramente.": "A",
    "Às vezes.": "B",
    "Frequentemente.": "B",
    "Sempre.": "B",
}
ROTULOS_GRUPOS = {
    "A": "Revisa detalhadamente",
    "B": "Sem revisão detalhada",
}
FAIXAS_Q19 = ["Nunca / Raramente", "Às vezes", "Frequentemente / Sempre"]
MAPA_Q19 = {
    "Nunca.": FAIXAS_Q19[0],
    "Raramente.": FAIXAS_Q19[0],
    "Às vezes.": FAIXAS_Q19[1],
    "Frequentemente.": FAIXAS_Q19[2],
    "Sempre.": FAIXAS_Q19[2],
}
CORES = ["#1f4e78", "#bdc3c7", "#c0392b"]


def localizar_coluna(df, numero):
    prefixo = re.compile(rf"^\s*{numero}\s*[\).\-:]")
    for coluna in df.columns:
        if prefixo.match(str(coluna)):
            return coluna

    raise ValueError(f"Não foi possível localizar a coluna da questão {numero}.")


def validar_respostas(respostas, mapeamento, numero_questao):
    desconhecidas = set(respostas.dropna().unique()) - mapeamento.keys()
    if desconhecidas:
        raise ValueError(
            f"Há respostas da Q{numero_questao} sem classificação: "
            + ", ".join(sorted(desconhecidas))
        )


def gerar_grafico(caminho_csv=ARQUIVO_CSV, caminho_saida=ARQUIVO_SAIDA):
    caminho_csv = Path(caminho_csv)
    caminho_saida = Path(caminho_saida)
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    col_q11 = localizar_coluna(df, 11)
    col_q19 = localizar_coluna(df, 19)

    dados = df[[col_q11, col_q19]].copy()
    for coluna in dados.columns:
        dados[coluna] = dados[coluna].astype("string").str.strip()
    dados = dados.dropna()
    if dados.empty:
        raise ValueError("Não há respostas válidas para o cruzamento entre Q11 e Q19.")

    validar_respostas(dados[col_q11], MAPA_Q11, 11)
    validar_respostas(dados[col_q19], MAPA_Q19, 19)
    dados["Grupo_Q11"] = dados[col_q11].map(MAPA_Q11)
    dados["Q19_faixa"] = dados[col_q19].map(MAPA_Q19)

    grupos_presentes = [
        grupo for grupo in GRUPOS_Q11 if grupo in set(dados["Grupo_Q11"])
    ]
    tabela = pd.crosstab(dados["Grupo_Q11"], dados["Q19_faixa"]).reindex(
        index=grupos_presentes,
        columns=FAIXAS_Q19,
        fill_value=0,
    )
    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    rotulos_grupos = [
        f"Grupo {grupo}\n"
        f"({ROTULOS_GRUPOS[grupo]}, n={tabela.loc[grupo].sum()})"
        for grupo in grupos_presentes
    ]

    fig, ax = plt.subplots(figsize=(8.8, 3.3), dpi=300)
    esquerda = np.zeros(len(grupos_presentes))
    posicoes_y = np.arange(len(grupos_presentes))

    for i, faixa in enumerate(FAIXAS_Q19):
        valores = percentuais[faixa].to_numpy()
        contagens = tabela[faixa].to_numpy()
        ax.barh(
            posicoes_y,
            valores,
            left=esquerda,
            height=0.52,
            label=faixa,
            color=CORES[i],
            edgecolor="white",
            linewidth=1.0,
        )
        for j, (valor, contagem) in enumerate(zip(valores, contagens)):
            if valor >= 5:
                cor_texto = "white" if i in (0, 2) else "#1a1a1a"
                valor_formatado = f"{valor:.2f}".replace(".", ",")
                ax.text(
                    esquerda[j] + valor / 2,
                    posicoes_y[j],
                    f"{valor_formatado}%\n(n={contagem})",
                    va="center",
                    ha="center",
                    color=cor_texto,
                    fontsize=9,
                    fontweight="bold",
                )
        esquerda += valores

    ax.set_yticks(posicoes_y)
    ax.set_yticklabels(rotulos_grupos, fontsize=9.8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proporção de participantes (%)", fontsize=10)
    ax.set_title(
        "Dificuldade de desenvolver soluções sem auxílio da IA por grupo de revisão",
        fontsize=11,
        fontweight="bold",
        pad=42,
    )
    ax.legend(
        title="Frequência de dificuldade sem assistência da IA (Questão 19)",
        bbox_to_anchor=(0.5, 1.18),
        loc="center",
        ncol=len(FAIXAS_Q19),
        frameon=False,
        fontsize=9,
        title_fontsize=9.2,
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
