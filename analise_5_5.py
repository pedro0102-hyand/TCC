
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

candidatos = [
    BASE_DIR / "data.csv",
    BASE_DIR / "dados" / "data.csv",
    BASE_DIR / "data(2).csv",
    BASE_DIR / "dados" / "data(2).csv",
]

ARQUIVO = next((p for p in candidatos if p.exists()), None)

if ARQUIVO is None:
    raise FileNotFoundError(
        "Não encontrei data.csv. Coloque o arquivo na pasta "
        "do projeto ou dentro de dados/."
    )

SAIDA = BASE_DIR / "resultados" / "analise_5_5"
SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")

# Identificação das colunas
def localizar_coluna(numero):
    prefixo = f"{numero})"
    colunas = [
        c for c in df.columns
        if str(c).strip().startswith(prefixo)
    ]
    if not colunas:
        raise KeyError(f"Questão {numero} não encontrada.")
    return colunas[0]

q16 = localizar_coluna(16)
q17 = localizar_coluna(17)

dados = df[[q16, q17]].copy()

for coluna in [q16, q17]:
    dados[coluna] = dados[coluna].astype("string").str.strip()
    dados[coluna] = dados[coluna].replace("", pd.NA)

# ============================================================
# FUNÇÃO DE DISTRIBUIÇÃO
# ============================================================

def analisar_distribuicao(serie, nome, ordem):
    respostas = serie.dropna()
    frequencias = respostas.value_counts().reindex(
        ordem, fill_value=0
    )
    percentuais = frequencias / len(respostas) * 100

    tabela = pd.DataFrame({
        "Frequência": frequencias,
        "Percentual (%)": percentuais.round(2)
    })

    print(f"\n{nome}")
    print(f"Respostas válidas: {len(respostas)}")
    print(f"Respostas ausentes: {len(df) - len(respostas)}")
    print(tabela.to_string())

    tabela.to_csv(
        SAIDA / f"{nome.lower()}_frequencias.csv",
        encoding="utf-8-sig"
    )

    return tabela

# ============================================================
# ORDENS DAS ALTERNATIVAS
# ============================================================

ordem_q16 = [
    "Diminuiu muito.",
    "Diminuiu.",
    "Permaneceu igual.",
    "Aumentou.",
    "Aumentou muito.",
    "Não sei avaliar."
]

ordem_q17 = [
    "Discordo totalmente.",
    "Discordo.",
    "Nem concordo nem discordo.",
    "Concordo.",
    "Concordo totalmente."
]

# ============================================================
# Q16 — DEPENDÊNCIA
# ============================================================

print("=" * 70)
print("ANÁLISE 5.5.1 — DEPENDÊNCIA E RETENÇÃO DO CONHECIMENTO")
print("=" * 70)

print(f"Arquivo: {ARQUIVO}")
print(f"Participantes: {len(df)}")

tab16 = analisar_distribuicao(
    dados[q16], "Q16", ordem_q16
)

# ============================================================
# Q17 — RETENÇÃO DO CONHECIMENTO
# ============================================================

tab17 = analisar_distribuicao(
    dados[q17], "Q17", ordem_q17
)

# ============================================================
# CRUZAMENTO Q16 × Q17
# ============================================================

cruzamento = pd.crosstab(dados[q16], dados[q17])

cruzamento = cruzamento.reindex(
    index=ordem_q16,
    columns=ordem_q17,
    fill_value=0
)

percentuais = cruzamento.div(
    cruzamento.sum(axis=1), axis=0
) * 100

print("\n" + "=" * 70)
print("CRUZAMENTO Q16 × Q17 — FREQUÊNCIAS ABSOLUTAS")
print("=" * 70)
print(cruzamento.to_string())

print("\n" + "=" * 70)
print("CRUZAMENTO Q16 × Q17 — PERCENTUAIS POR LINHA")
print("=" * 70)
print(percentuais.round(2).to_string())

cruzamento.to_csv(
    SAIDA / "q16_q17_contagens.csv",
    encoding="utf-8-sig"
)

percentuais.round(2).to_csv(
    SAIDA / "q16_q17_percentuais.csv",
    encoding="utf-8-sig"
)

# ============================================================
# GRÁFICO Q16
# ============================================================

def grafico_distribuicao(tabela, titulo, arquivo):
    plt.figure(figsize=(10, 5.5))

    valores = tabela["Frequência"]
    barras = plt.bar(
        range(len(valores)),
        valores.values
    )

    plt.xticks(
        range(len(valores)),
        valores.index,
        rotation=25,
        ha="right"
    )

    plt.ylabel("Quantidade de participantes")
    plt.title(titulo)
    plt.grid(axis="y", alpha=0.25)

    for barra, valor in zip(barras, valores.values):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.2,
            str(int(valor)),
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(SAIDA / arquivo, dpi=300, bbox_inches="tight")
    plt.close()

grafico_distribuicao(
    tab16,
    "Q16 — Percepção de dependência das ferramentas de IA",
    "q16_dependencia.png"
)

grafico_distribuicao(
    tab17,
    "Q17 — Percepção de dificuldade na retenção do conhecimento",
    "q17_retencao.png"
)

# ============================================================
# GRÁFICO DO CRUZAMENTO — PERCENTUAIS POR LINHA
# ============================================================

plt.figure(figsize=(12, 6))

imagem = plt.imshow(
    percentuais.values,
    aspect="auto",
    cmap="Blues",
    vmin=0,
    vmax=100
)

plt.colorbar(imagem, label="Percentual dentro da categoria de Q16")

plt.xticks(
    range(len(ordem_q17)),
    ordem_q17,
    rotation=25,
    ha="right"
)

plt.yticks(
    range(len(ordem_q16)),
    ordem_q16
)

plt.title("Q16 × Q17 — Dependência percebida e retenção do conhecimento")

for i in range(len(ordem_q16)):
    for j in range(len(ordem_q17)):
        valor = percentuais.iloc[i, j]
        quantidade = cruzamento.iloc[i, j]

        plt.text(
            j, i,
            f"{valor:.1f}%\n(n={quantidade})",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()
plt.savefig(
    SAIDA / "q16_q17_cruzamento.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

for arquivo in sorted(SAIDA.iterdir()):
    print(arquivo)

print("\nAnálise concluída.")
