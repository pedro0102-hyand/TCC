from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_CSV = BASE_DIR / "dados" / "data.csv"
PASTA_SAIDA = BASE_DIR / "resultados"

ORDEM_Q16 = [
    "Aumentou muito",
    "Aumentou",
    "Permaneceu igual / outros",
]
MAPA_Q16 = {
    "Aumentou muito.": ORDEM_Q16[0],
    "Aumentou.": ORDEM_Q16[1],
    "Permaneceu igual.": ORDEM_Q16[2],
    "Diminuiu.": ORDEM_Q16[2],
    "Não sei avaliar.": ORDEM_Q16[2],
}
ROTULOS_Q16 = {
    ORDEM_Q16[0]: "Aumentou muito",
    ORDEM_Q16[1]: "Aumentou",
    ORDEM_Q16[2]: "Permaneceu igual / outros",
}

ORDEM_Q17 = [
    "Discordo / Disc. totalmente",
    "Nem concordo nem discordo",
    "Concordo / Conc. totalmente",
]
MAPA_Q17 = {
    "Discordo.": ORDEM_Q17[0],
    "Discordo totalmente.": ORDEM_Q17[0],
    "Nem concordo nem discordo.": ORDEM_Q17[1],
    "Concordo.": ORDEM_Q17[2],
    "Concordo totalmente.": ORDEM_Q17[2],
}

ORDEM_EVOLUCAO = [
    "Piorou / Piorou muito",
    "Permaneceu semelhante",
    "Melhorou / Melhorou muito",
]
MAPA_EVOLUCAO = {
    "Piorou.": ORDEM_EVOLUCAO[0],
    "Piorou muito.": ORDEM_EVOLUCAO[0],
    "Permaneceu semelhante.": ORDEM_EVOLUCAO[1],
    "Melhorou.": ORDEM_EVOLUCAO[2],
    "Melhorou muito.": ORDEM_EVOLUCAO[2],
}


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


def criar_tabela_cruzada(
    df,
    coluna_linha,
    coluna_coluna,
    mapa_linha,
    mapa_coluna,
    ordem_linha,
    ordem_coluna,
    numero_linha,
    numero_coluna,
):
    dados = df[[coluna_linha, coluna_coluna]].copy()
    for coluna in dados.columns:
        dados[coluna] = dados[coluna].astype("string").str.strip()
    dados = dados.dropna()
    if dados.empty:
        raise ValueError(
            f"Não há respostas válidas para o cruzamento entre "
            f"Q{numero_linha} e Q{numero_coluna}."
        )

    validar_respostas(dados[coluna_linha], mapa_linha, numero_linha)
    validar_respostas(dados[coluna_coluna], mapa_coluna, numero_coluna)
    dados["linha"] = dados[coluna_linha].map(mapa_linha)
    dados["coluna"] = dados[coluna_coluna].map(mapa_coluna)

    presentes = set(dados["linha"])
    ordem_presente = [categoria for categoria in ordem_linha if categoria in presentes]
    tabela = pd.crosstab(dados["linha"], dados["coluna"]).reindex(
        index=ordem_presente,
        columns=ordem_coluna,
        fill_value=0,
    )
    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    return tabela, percentuais, ordem_presente


def plot_barras_agrupadas(
    tabela_percentuais,
    tabela_contagens,
    rotulos_x,
    ordem_linha,
    ordem_coluna,
    cores,
    titulo,
    titulo_legenda,
    caminho_saida,
):
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=300)
    posicoes_x = np.arange(len(rotulos_x))
    largura = 0.25
    quantidade_series = len(ordem_coluna)

    for i, categoria in enumerate(ordem_coluna):
        valores = tabela_percentuais[categoria].to_numpy()
        contagens = tabela_contagens[categoria].to_numpy()
        deslocamento = (i - (quantidade_series - 1) / 2) * largura
        barras = ax.bar(
            posicoes_x + deslocamento,
            valores,
            width=largura * 0.92,
            label=categoria,
            color=cores[i],
            edgecolor="white",
            linewidth=0.8,
        )
        for barra, valor, contagem in zip(barras, valores, contagens):
            if valor > 0:
                valor_formatado = f"{valor:.1f}".replace(".", ",")
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 1.5,
                    f"{valor_formatado}%\n(n={contagem})",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color="#1a1a1a",
                )

    ax.set_xticks(posicoes_x)
    ax.set_xticklabels(rotulos_x, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Proporção dentro do grupo (%)", fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight="bold", pad=42)
    ax.legend(
        title=titulo_legenda,
        bbox_to_anchor=(0.5, 1.13),
        loc="center",
        ncol=len(ordem_coluna),
        frameon=False,
        fontsize=8.8,
        title_fontsize=9.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")


def rotulos_com_contagem(tabela, ordem, nomes):
    return [
        f"{nomes[categoria]}\n(n={tabela.loc[categoria].sum()})"
        for categoria in ordem
    ]


def main():
    if not ARQUIVO_CSV.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {ARQUIVO_CSV}")

    df = pd.read_csv(ARQUIVO_CSV, encoding="utf-8-sig")
    col_q16 = localizar_coluna(df, 16)
    col_q17 = localizar_coluna(df, 17)
    col_q20 = localizar_coluna(df, 20)
    col_q21 = localizar_coluna(df, 21)

    tabela1, percentuais1, ordem1 = criar_tabela_cruzada(
        df,
        col_q16,
        col_q17,
        MAPA_Q16,
        MAPA_Q17,
        ORDEM_Q16,
        ORDEM_Q17,
        16,
        17,
    )
    plot_barras_agrupadas(
        percentuais1,
        tabela1,
        rotulos_com_contagem(tabela1, ordem1, ROTULOS_Q16),
        ordem1,
        ORDEM_Q17,
        ["#1f4e78", "#95a5a6", "#c0392b"],
        "Dificuldade de retenção segundo a evolução da dependência de IAG",
        "O uso frequente de IA dificulta a retenção do conhecimento (Questão 17)",
        PASTA_SAIDA / "q16_q17_cruzamento.png",
    )

    tabela2, percentuais2, ordem2 = criar_tabela_cruzada(
        df,
        col_q20,
        col_q21,
        MAPA_EVOLUCAO,
        MAPA_EVOLUCAO,
        ORDEM_EVOLUCAO,
        ORDEM_EVOLUCAO,
        20,
        21,
    )
    plot_barras_agrupadas(
        percentuais2,
        tabela2,
        rotulos_com_contagem(tabela2, ordem2, {categoria: categoria for categoria in ordem2}),
        ordem2,
        ORDEM_EVOLUCAO,
        ["#c0392b", "#95a5a6", "#1f4e78"],
        "Compreensão de conceitos segundo a resolução autônoma sem IA",
        "Evolução da capacidade de compreender conceitos (Questão 21)",
        PASTA_SAIDA / "q20_q21_cruzamento.png",
    )


if __name__ == "__main__":
    main()
