
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

PASTA_SAIDA = BASE_DIR / "resultados" / "analise_5_5_3"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO)

# Q22 corresponde à coluna de índice 22.
q22 = df.iloc[:, 22].astype("string").str.strip()

ordem = [
    "Nunca.",
    "Raramente.",
    "Às vezes.",
    "Frequentemente.",
    "Sempre."
]

# ============================================================
# DISTRIBUIÇÃO DE FREQUÊNCIAS
# ============================================================

frequencias = q22.value_counts().reindex(
    ordem,
    fill_value=0
)

total = q22.notna().sum()
percentuais = (frequencias / total * 100).round(2)

resultado = pd.DataFrame({
    "Frequência": frequencias,
    "Percentual (%)": percentuais
})

print("=" * 70)
print("ANÁLISE 5.5.3 — CONSULTA A FONTES EXTERNAS")
print("=" * 70)

print(f"Arquivo: {ARQUIVO}")
print(f"Participantes: {len(df)}")
print(f"Respostas válidas: {total}")
print(f"Respostas ausentes: {q22.isna().sum()}")

print("\nQ22 — DISTRIBUIÇÃO DAS RESPOSTAS")
print(resultado.to_string())

# ============================================================
# VERIFICAÇÃO DE RESPOSTAS NÃO CONTABILIZADAS
# ============================================================

respostas_nao_classificadas = q22[
    q22.notna() & ~q22.isin(ordem)
]

if not respostas_nao_classificadas.empty:
    print("\nATENÇÃO: Foram encontradas respostas não classificadas:")
    print(respostas_nao_classificadas.value_counts().to_string())

    raise ValueError(
        "Existem respostas fora das categorias previstas. "
        "Verifique os valores antes de continuar."
    )

# ============================================================
# AGRUPAMENTO DAS RESPOSTAS
# ============================================================

frequencia_regular = (
    frequencias["Frequentemente."] +
    frequencias["Sempre."]
)

frequencia_ocasional = frequencias["Às vezes."]

sem_consulta_frequente = (
    frequencias["Nunca."] +
    frequencias["Raramente."]
)

print("\n" + "=" * 70)
print("AGRUPAMENTO DAS RESPOSTAS")
print("=" * 70)

print(
    f"Frequentemente ou sempre: "
    f"{frequencia_regular} "
    f"({frequencia_regular / total * 100:.2f}%)"
)

print(
    f"Às vezes: "
    f"{frequencia_ocasional} "
    f"({frequencia_ocasional / total * 100:.2f}%)"
)

print(
    f"Nunca ou raramente: "
    f"{sem_consulta_frequente} "
    f"({sem_consulta_frequente / total * 100:.2f}%)"
)

# ============================================================
# EXPORTAÇÃO DOS RESULTADOS
# ============================================================

resultado.to_csv(
    PASTA_SAIDA / "q22_frequencias.csv",
    encoding="utf-8-sig"
)

agrupamento = pd.DataFrame({
    "Grupo": [
        "Frequentemente ou sempre",
        "Às vezes",
        "Nunca ou raramente"
    ],
    "Frequência": [
        frequencia_regular,
        frequencia_ocasional,
        sem_consulta_frequente
    ]
})

agrupamento["Percentual (%)"] = (
    agrupamento["Frequência"] / total * 100
).round(2)

agrupamento.to_csv(
    PASTA_SAIDA / "q22_agrupamento.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# GRÁFICO DE DISTRIBUIÇÃO
# ============================================================

dados = resultado.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 5.5))

barras = ax.barh(
    dados.index,
    dados["Frequência"]
)

ax.set_title(
    "Frequência de consulta a fontes externas durante o aprendizado",
    fontsize=13,
    pad=15
)

ax.set_xlabel("Número de participantes")
ax.set_ylabel("Frequência de consulta")

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
    PASTA_SAIDA / "q22_fontes_externas.png",
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
