
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "dados/data.csv"
PASTA_SAIDA = Path("resultados/analise_5_5_4")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# LEITURA DOS DADOS
# ============================================================

df = pd.read_csv(ARQUIVO)

q23 = df.iloc[:, 23].dropna().astype(str).str.strip()
q24 = df.iloc[:, 24].dropna().astype(str).str.strip()

dados = pd.DataFrame({
    "Q23": df.iloc[:, 23],
    "Q24": df.iloc[:, 24]
}).dropna()

dados["Q23"] = dados["Q23"].astype(str).str.strip()
dados["Q24"] = dados["Q24"].astype(str).str.strip()

# ============================================================
# ORDENAÇÃO DAS RESPOSTAS
# ============================================================

ordem_q23 = [
    "Nunca.",
    "Raramente.",
    "Às vezes.",
    "Frequentemente.",
    "Sempre."
]

ordem_q24 = [
    "Discordo totalmente.",
    "Discordo.",
    "Nem concordo nem discordo.",
    "Concordo.",
    "Concordo totalmente."
]

# Verificação para evitar perda silenciosa de categorias
categorias_q23 = set(q23.unique())
categorias_q24 = set(q24.unique())

inesperadas_q23 = categorias_q23 - set(ordem_q23)
inesperadas_q24 = categorias_q24 - set(ordem_q24)

if inesperadas_q23:
    raise ValueError(f"Categorias não classificadas em Q23: {inesperadas_q23}")

if inesperadas_q24:
    raise ValueError(f"Categorias não classificadas em Q24: {inesperadas_q24}")

# ============================================================
# FUNÇÃO DE FREQUÊNCIAS
# ============================================================

def analisar_frequencias(serie, ordem, nome):
    frequencias = serie.value_counts().reindex(ordem, fill_value=0)
    percentuais = (frequencias / len(serie) * 100).round(2)

    tabela = pd.DataFrame({
        "Frequência": frequencias,
        "Percentual (%)": percentuais
    })

    print(f"\n{'=' * 60}")
    print(f"DISTRIBUIÇÃO {nome}")
    print(f"{'=' * 60}")
    print(tabela.to_string())

    tabela.to_csv(
        PASTA_SAIDA / f"{nome.lower()}_frequencias.csv",
        encoding="utf-8-sig"
    )

    return frequencias

freq_q23 = analisar_frequencias(q23, ordem_q23, "Q23")
freq_q24 = analisar_frequencias(q24, ordem_q24, "Q24")

# ============================================================
# GRÁFICOS INDIVIDUAIS
# ============================================================

def gerar_grafico(frequencias, titulo, arquivo, ylabel):
    plt.figure(figsize=(10, 6))

    ax = frequencias.plot(
        kind="bar",
        figsize=(10, 6),
        edgecolor="black"
    )

    plt.title(titulo, fontsize=13, pad=15)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for barra in ax.patches:
        altura = barra.get_height()
        ax.annotate(
            f"{int(altura)}",
            (barra.get_x() + barra.get_width() / 2, altura),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points"
        )

    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / arquivo, dpi=300, bbox_inches="tight")
    plt.close()


gerar_grafico(
    freq_q23,
    "Frequência de aprendizagem de conceitos com IA",
    "q23_aprendizado_conceitos.png",
    "Número de participantes"
)

gerar_grafico(
    freq_q24,
    "Percepção da IA como facilitadora da aprendizagem",
    "q24_facilitadora_aprendizado.png",
    "Número de participantes"
)

# ============================================================
# CRUZAMENTO Q23 x Q24
# ============================================================

cruzamento = pd.crosstab(
    dados["Q23"],
    dados["Q24"]
).reindex(
    index=ordem_q23,
    columns=ordem_q24,
    fill_value=0
)

percentuais_linha = (
    cruzamento.div(cruzamento.sum(axis=1), axis=0) * 100
).round(2)

print(f"\n{'=' * 60}")
print("CRUZAMENTO Q23 x Q24 — FREQUÊNCIAS")
print(f"{'=' * 60}")
print(cruzamento.to_string())

print(f"\n{'=' * 60}")
print("CRUZAMENTO Q23 x Q24 — PERCENTUAIS POR LINHA")
print(f"{'=' * 60}")
print(percentuais_linha.to_string())

cruzamento.to_csv(
    PASTA_SAIDA / "q23_q24_frequencias.csv",
    encoding="utf-8-sig"
)

percentuais_linha.to_csv(
    PASTA_SAIDA / "q23_q24_percentuais_linha.csv",
    encoding="utf-8-sig"
)

# ============================================================
# HEATMAP
# ============================================================

plt.figure(figsize=(12, 7))

sns.heatmap(
    percentuais_linha,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    linewidths=0.5,
    cbar_kws={"label": "Percentual dentro da categoria Q23"}
)

plt.title(
    "Relação entre frequência de aprendizagem com IA (Q23) "
    "e percepção de facilitação da aprendizagem (Q24)",
    fontsize=12,
    pad=15
)

plt.xlabel("Percepção sobre a facilitação da aprendizagem (Q24)")
plt.ylabel("Frequência de aprendizagem de conceitos com IA (Q23)")
plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(
    PASTA_SAIDA / "q23_q24_cruzamento.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# AGRUPAMENTOS COMPLEMENTARES
# ============================================================

print(f"\n{'=' * 60}")
print("AGRUPAMENTOS COMPLEMENTARES")
print(f"{'=' * 60}")

q23_recorrente = q23.isin(["Frequentemente", "Sempre"]).sum()
q23_ocasional = q23.eq("Às vezes").sum()
q23_baixa = q23.isin(["Nunca", "Raramente"]).sum()

q24_concordancia = q24.isin([
    "Concordo",
    "Concordo totalmente"
]).sum()

q24_neutralidade = q24.eq("Nem concordo nem discordo").sum()

q24_discordancia = q24.isin([
    "Discordo",
    "Discordo totalmente"
]).sum()

for nome, valor, total in [
    ("Q23 — Frequentemente/Sempre", q23_recorrente, len(q23)),
    ("Q23 — Às vezes", q23_ocasional, len(q23)),
    ("Q23 — Nunca/Raramente", q23_baixa, len(q23)),
    ("Q24 — Concordo/Concordo totalmente", q24_concordancia, len(q24)),
    ("Q24 — Neutros", q24_neutralidade, len(q24)),
    ("Q24 — Discordo/Discordo totalmente", q24_discordancia, len(q24))
]:
    print(f"{nome}: {valor}/{total} ({valor / total * 100:.2f}%)")

print(f"\nTotal de respostas Q23: {len(q23)}")
print(f"Total de respostas Q24: {len(q24)}")
print(f"Total de respostas válidas no cruzamento: {len(dados)}")

print(f"\nArquivos salvos em: {PASTA_SAIDA.resolve()}")
