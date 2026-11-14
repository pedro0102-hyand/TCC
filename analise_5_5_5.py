
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "dados/data.csv"

PASTA_SAIDA = Path("resultados/analise_5_5_5")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# LEITURA DOS DADOS
# ============================================================

df = pd.read_csv(ARQUIVO)

def normalizar(serie):
    return (
        serie.dropna()
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .str.strip()
    )

# Recuperação das questões
dados = pd.DataFrame({
    "Q17": df.iloc[:, 17],
    "Q19": df.iloc[:, 19],
    "Q20": df.iloc[:, 20],
    "Q23": df.iloc[:, 23],
    "Q24": df.iloc[:, 24]
})

for coluna in dados.columns:
    dados[coluna] = normalizar(dados[coluna])

# ============================================================
# ORDENAÇÃO DAS CATEGORIAS
# ============================================================

ordens = {
    "Q17": [
        "Discordo totalmente",
        "Discordo",
        "Nem concordo nem discordo",
        "Concordo",
        "Concordo totalmente"
    ],

    "Q19": [
        "Raramente",
        "Às vezes",
        "Frequentemente",
        "Sempre"
    ],

    "Q20": [
        "Piorou muito",
        "Piorou",
        "Permaneceu semelhante",
        "Melhorou",
        "Melhorou muito",
        "Não sei avaliar"
    ],

    "Q23": [
        "Nunca",
        "Raramente",
        "Às vezes",
        "Frequentemente",
        "Sempre"
    ],

    "Q24": [
        "Discordo totalmente",
        "Discordo",
        "Nem concordo nem discordo",
        "Concordo",
        "Concordo totalmente"
    ]
}

# ============================================================
# VALIDAÇÃO DAS CATEGORIAS
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÃO DOS DADOS")
print("=" * 70)

for coluna, ordem in ordens.items():

    encontradas = set(dados[coluna].dropna().unique())
    inesperadas = encontradas - set(ordem)

    print(f"\n{coluna}:")
    print(f"Respostas válidas: {dados[coluna].notna().sum()}")
    print(f"Respostas ausentes: {dados[coluna].isna().sum()}")

    if inesperadas:
        raise ValueError(
            f"Categorias não classificadas em {coluna}: {inesperadas}"
        )

    print("Todas as categorias foram reconhecidas.")

# ============================================================
# CLASSIFICAÇÃO DA Q24
# ============================================================

dados["Grupo_Q24"] = dados["Q24"].map({
    "Concordo": "Percepção positiva",
    "Concordo totalmente": "Percepção positiva",
    "Nem concordo nem discordo": "Sem concordância",
    "Discordo": "Sem concordância",
    "Discordo totalmente": "Sem concordância"
})

ordem_grupos = [
    "Percepção positiva",
    "Sem concordância"
]

print("\n" + "=" * 70)
print("DISTRIBUIÇÃO DOS GRUPOS Q24")
print("=" * 70)

grupos = dados["Grupo_Q24"].value_counts().reindex(
    ordem_grupos,
    fill_value=0
)

total_grupos = dados["Grupo_Q24"].notna().sum()

for grupo, quantidade in grupos.items():

    percentual = (
        quantidade / total_grupos * 100
        if total_grupos > 0 else 0
    )

    print(
        f"{grupo}: {quantidade} "
        f"({percentual:.2f}%)"
    )

# ============================================================
# FUNÇÃO DE ANÁLISE E VISUALIZAÇÃO
# ============================================================

def analisar_cruzamento(variavel, titulo, arquivo):

    # Seleciona apenas respostas válidas nas duas questões
    base = dados[["Grupo_Q24", variavel]].dropna()

    # --------------------------------------------------------
    # TABELA DE FREQUÊNCIAS
    # --------------------------------------------------------

    tabela = pd.crosstab(
        base["Grupo_Q24"],
        base[variavel]
    ).reindex(
        index=ordem_grupos,
        columns=ordens[variavel],
        fill_value=0
    )

    # --------------------------------------------------------
    # PERCENTUAIS POR GRUPO
    # --------------------------------------------------------

    percentuais = (
        tabela.div(
            tabela.sum(axis=1).replace(0, float("nan")),
            axis=0
        ) * 100
    ).fillna(0).round(2)

    # --------------------------------------------------------
    # EXIBIÇÃO NO TERMINAL
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(f"{titulo} — FREQUÊNCIAS")
    print("=" * 75)

    print(tabela.to_string())

    print("\n" + "=" * 75)
    print(f"{titulo} — PERCENTUAIS POR GRUPO Q24")
    print("=" * 75)

    print(percentuais.to_string())

    # --------------------------------------------------------
    # EXPORTAÇÃO DAS TABELAS
    # --------------------------------------------------------

    tabela.to_csv(
        PASTA_SAIDA / f"{arquivo}_frequencias.csv",
        encoding="utf-8-sig"
    )

    percentuais.to_csv(
        PASTA_SAIDA / f"{arquivo}_percentuais.csv",
        encoding="utf-8-sig"
    )

    # ========================================================
    # GRÁFICO DE BARRAS HORIZONTAIS EMPILHADAS A 100%
    # ========================================================

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Paleta automática
    cores = plt.get_cmap("Set2").colors

    esquerda = [0] * len(ordem_grupos)

    for i, categoria in enumerate(ordens[variavel]):

        valores = percentuais[categoria].values

        barras = ax.barh(
            range(len(ordem_grupos)),
            valores,
            left=esquerda,
            label=categoria,
            color=cores[i % len(cores)],
            edgecolor="white",
            linewidth=1.2,
            height=0.55
        )

        # Percentuais dentro dos segmentos
        for j, barra in enumerate(barras):

            largura = barra.get_width()

            if largura >= 8:

                ax.text(
                    barra.get_x() + largura / 2,
                    barra.get_y() + barra.get_height() / 2,
                    f"{largura:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9
                )

        esquerda = [
            esquerda[j] + valores[j]
            for j in range(len(esquerda))
        ]

    # --------------------------------------------------------
    # RÓTULOS COM TAMANHO DOS GRUPOS
    # --------------------------------------------------------

    tamanhos = tabela.sum(axis=1)

    rotulos = [
        f"{grupo} (n={tamanhos[grupo]})"
        for grupo in ordem_grupos
    ]

    ax.set_yticks(range(len(ordem_grupos)))
    ax.set_yticklabels(rotulos, fontsize=10)

    # --------------------------------------------------------
    # FORMATAÇÃO
    # --------------------------------------------------------

    ax.set_xlim(0, 100)

    ax.set_xlabel(
        "Percentual dentro de cada grupo (%)",
        fontsize=10
    )

    ax.set_title(
        titulo,
        fontsize=13,
        pad=18
    )

    ax.legend(
        title="Categorias de resposta",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # --------------------------------------------------------
    # SALVAMENTO
    # --------------------------------------------------------

    plt.savefig(
        PASTA_SAIDA / f"{arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nGráfico salvo: "
        f"{PASTA_SAIDA / f'{arquivo}.png'}"
    )

    return tabela, percentuais

# ============================================================
# Q24 x Q17 — RETENÇÃO DO CONHECIMENTO
# ============================================================

tabela_17, perc_17 = analisar_cruzamento(
    "Q17",
    "Percepção da IA e dificuldade de retenção do conhecimento",
    "q24_q17_retencao"
)

# ============================================================
# Q24 x Q19 — DIFICULDADE DE DESENVOLVER SEM IA
# ============================================================

tabela_19, perc_19 = analisar_cruzamento(
    "Q19",
    "Percepção da IA e dificuldade de desenvolver sem IA",
    "q24_q19_autonomia"
)

# ============================================================
# Q24 x Q20 — EVOLUÇÃO DA RESOLUÇÃO AUTÔNOMA
# ============================================================

tabela_20, perc_20 = analisar_cruzamento(
    "Q20",
    "Percepção da IA e evolução da resolução autônoma",
    "q24_q20_resolucao"
)

# ============================================================
# Q24 x Q23 — FREQUÊNCIA DE APRENDIZAGEM
# ============================================================

tabela_23, perc_23 = analisar_cruzamento(
    "Q23",
    "Percepção da IA e frequência de aprendizagem de conceitos",
    "q24_q23_aprendizado"
)

# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 75)
print("RESUMO FINAL DA ANÁLISE")
print("=" * 75)

print("\nTamanho dos grupos Q24:")

for grupo in ordem_grupos:

    n = (dados["Grupo_Q24"] == grupo).sum()

    print(f"{grupo}: {n} participantes")

print("\nRespostas válidas em cada cruzamento:")

for variavel in ["Q17", "Q19", "Q20", "Q23"]:

    n = dados[["Grupo_Q24", variavel]].dropna().shape[0]

    print(f"Q24 x {variavel}: {n} participantes")

print("\nArquivos gerados:")

for arquivo in sorted(PASTA_SAIDA.iterdir()):
    print(f"- {arquivo.name}")

print("\n" + "=" * 75)
print("ANÁLISE CONCLUÍDA")
print("=" * 75)
