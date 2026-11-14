
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POSSIVEIS_ARQUIVOS = [
    BASE_DIR / "dados" / "data.csv",
    BASE_DIR / "data.csv",
    BASE_DIR / "data(2).csv",
]

ARQUIVO = next(
    (arquivo for arquivo in POSSIVEIS_ARQUIVOS if arquivo.exists()),
    None
)

if ARQUIVO is None:
    raise FileNotFoundError(
        "Não foi possível localizar o arquivo data.csv."
    )

PASTA_SAIDA = BASE_DIR / "resultados" / "analise_5_5_2"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO)

# Q20 e Q21 correspondem às colunas de índice 20 e 21.
q20 = df.iloc[:, 20].astype("string").str.strip()
q21 = df.iloc[:, 21].astype("string").str.strip()

ordem = [
    "Piorou muito.",
    "Piorou.",
    "Permaneceu semelhante.",
    "Melhorou.",
    "Melhorou muito."
]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def analisar_distribuicao(serie, nome, ordem_respostas):
    frequencias = serie.value_counts().reindex(
        ordem_respostas,
        fill_value=0
    )

    total = serie.notna().sum()
    percentuais = (frequencias / total * 100).round(2)

    resultado = pd.DataFrame({
        "Frequência": frequencias,
        "Percentual (%)": percentuais
    })

    print("\n" + "=" * 70)
    print(nome)
    print("=" * 70)
    print(f"Respostas válidas: {total}")
    print(f"Respostas ausentes: {serie.isna().sum()}")
    print(resultado.to_string())

    resultado.to_csv(
        PASTA_SAIDA / f"{nome.lower()}_frequencias.csv",
        encoding="utf-8-sig"
    )

    return resultado


def gerar_grafico_distribuicao(resultado, titulo, nome_arquivo):
    dados = resultado.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    barras = ax.barh(
        dados.index,
        dados["Frequência"]
    )

    ax.set_title(titulo, fontsize=13, pad=15)
    ax.set_xlabel("Número de participantes")
    ax.set_ylabel("Resposta")

    ax.bar_label(
        barras,
        labels=[
            f"{int(valor)} ({percentual:.2f}%)"
            for valor, percentual in zip(
                dados["Frequência"],
                dados["Percentual (%)"]
            )
        ],
        padding=4
    )

    ax.set_xlim(
        0,
        max(dados["Frequência"].max() * 1.35, 1)
    )

    plt.tight_layout()
    plt.savefig(
        PASTA_SAIDA / nome_arquivo,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 1. DISTRIBUIÇÃO Q20
# ============================================================

resultado_q20 = analisar_distribuicao(
    q20,
    "Q20",
    ordem
)

gerar_grafico_distribuicao(
    resultado_q20,
    "Percepção sobre a capacidade de resolver problemas sem IA",
    "q20_resolucao_autonoma.png"
)


# ============================================================
# 2. DISTRIBUIÇÃO Q21
# ============================================================

resultado_q21 = analisar_distribuicao(
    q21,
    "Q21",
    ordem
)

gerar_grafico_distribuicao(
    resultado_q21,
    "Percepção sobre a evolução da compreensão de conceitos",
    "q21_compreensao_conceitos.png"
)


# ============================================================
# 3. AGRUPAMENTO: MELHORA, ESTABILIDADE E PIORA
# ============================================================

grupos = {
    "Piorou": [
        "Piorou muito.",
        "Piorou."
    ],
    "Permaneceu semelhante": [
        "Permaneceu semelhante."
    ],
    "Melhorou": [
        "Melhorou.",
        "Melhorou muito."
    ]
}

def agrupar_respostas(serie):
    resultado = pd.Series(index=serie.index, dtype="string")

    for grupo, alternativas in grupos.items():
        resultado.loc[serie.isin(alternativas)] = grupo

    return resultado


q20_grupos = agrupar_respostas(q20)
q21_grupos = agrupar_respostas(q21)

comparacao = pd.DataFrame({
    "Q20": q20_grupos.value_counts(),
    "Q21": q21_grupos.value_counts()
}).fillna(0).astype(int)

comparacao = comparacao.reindex([
    "Melhorou",
    "Permaneceu semelhante",
    "Piorou"
])

comparacao["Q20 (%)"] = (
    comparacao["Q20"] / q20.notna().sum() * 100
).round(2)

comparacao["Q21 (%)"] = (
    comparacao["Q21"] / q21.notna().sum() * 100
).round(2)

print("\n" + "=" * 70)
print("COMPARAÇÃO ENTRE MELHORA, ESTABILIDADE E PIORA")
print("=" * 70)
print(comparacao.to_string())

comparacao.to_csv(
    PASTA_SAIDA / "q20_q21_comparacao_agrupada.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 4. CRUZAMENTO Q20 × Q21
# ============================================================

cruzamento = pd.crosstab(q20, q21)

cruzamento = cruzamento.reindex(
    index=ordem,
    columns=ordem,
    fill_value=0
)

percentuais = cruzamento.div(
    cruzamento.sum(axis=1).replace(0, pd.NA),
    axis=0
) * 100

print("\n" + "=" * 70)
print("CRUZAMENTO Q20 × Q21 — FREQUÊNCIAS ABSOLUTAS")
print("=" * 70)
print(cruzamento.to_string())

print("\n" + "=" * 70)
print("CRUZAMENTO Q20 × Q21 — PERCENTUAIS POR LINHA")
print("=" * 70)
print(percentuais.round(2).to_string())

cruzamento.to_csv(
    PASTA_SAIDA / "q20_q21_contagens.csv",
    encoding="utf-8-sig"
)

percentuais.to_csv(
    PASTA_SAIDA / "q20_q21_percentuais.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 5. GRÁFICO DO CRUZAMENTO
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

imagem = ax.imshow(
    percentuais.fillna(0).values,
    cmap="Blues",
    aspect="auto",
    vmin=0,
    vmax=100
)

ax.set_xticks(range(len(ordem)))
ax.set_yticks(range(len(ordem)))

ax.set_xticklabels(ordem, rotation=30, ha="right")
ax.set_yticklabels(ordem)

ax.set_xlabel("Q21 — Compreensão de conceitos")
ax.set_ylabel("Q20 — Resolução de problemas sem IA")
ax.set_title(
    "Relação entre percepção de resolução autônoma "
    "e compreensão de conceitos",
    pad=15
)

for i in range(len(ordem)):
    for j in range(len(ordem)):
        valor = percentuais.iloc[i, j]

        if pd.notna(valor):
            ax.text(
                j,
                i,
                f"{valor:.1f}%\n(n={cruzamento.iloc[i, j]})",
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )

fig.colorbar(
    imagem,
    ax=ax,
    label="Percentual dentro da categoria de Q20"
)

plt.tight_layout()
plt.savefig(
    PASTA_SAIDA / "q20_q21_cruzamento.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

for arquivo in sorted(PASTA_SAIDA.iterdir()):
    print(arquivo)

print("\nAnálise concluída.")
