from pathlib import Path
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POSSIVEIS_ARQUIVOS = [
    BASE_DIR / "dados" / "data.csv",
    BASE_DIR / "dados" / "data(1).csv",
    BASE_DIR / "dados" / "data.csv",
    BASE_DIR / "dados" / "data(1).csv",
]

CAMINHO_CSV = next(
    (caminho for caminho in POSSIVEIS_ARQUIVOS if caminho.exists()),
    None
)

PASTA_SAIDA = BASE_DIR / "resultados" / "analise_5_4"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ORDEM_RESPOSTAS = [
    "Nunca.",
    "Raramente.",
    "Às vezes.",
    "Frequentemente.",
    "Sempre."
]

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar(texto):
    texto = str(texto).strip().lower()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def localizar_coluna(numero):
    padrao = re.compile(rf"^\s*{numero}\s*[\).\-:]")

    encontradas = [
        coluna for coluna in df.columns
        if padrao.search(str(coluna))
    ]

    if not encontradas:
        raise ValueError(
            f"Não foi possível localizar a questão Q{numero}."
        )

    if len(encontradas) > 1:
        print(
            f"Atenção: múltiplas colunas encontradas para Q{numero}. "
            f"Utilizando: {encontradas[0]}"
        )

    return encontradas[0]


def salvar_grafico(nome):
    caminho = PASTA_SAIDA / nome

    plt.tight_layout()
    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Gráfico salvo: {caminho.name}")


def ordenar_respostas(serie):
    categorias_presentes = serie.dropna().unique().tolist()

    ordem = [
        resposta for resposta in ORDEM_RESPOSTAS
        if resposta in categorias_presentes
    ]

    outras = [
        resposta for resposta in categorias_presentes
        if resposta not in ordem
    ]

    return ordem + sorted(outras)


def gerar_distribuicao(numero, coluna):
    frequencia = coluna.value_counts()

    ordem = ordenar_respostas(coluna)

    frequencia = frequencia.reindex(ordem, fill_value=0)

    percentual = frequencia / len(coluna.dropna()) * 100

    tabela = pd.DataFrame({
        "Frequência": frequencia,
        "Percentual": percentual
    })

    tabela.to_csv(
        PASTA_SAIDA / f"q{numero}_distribuicao.csv",
        encoding="utf-8-sig"
    )

    print(f"\nDISTRIBUIÇÃO Q{numero}")
    print(tabela.round(2).to_string())

    plt.figure(figsize=(9, 5))

    barras = plt.barh(
        frequencia.index,
        frequencia.values
    )

    plt.title(f"Distribuição das respostas — Q{numero}")
    plt.xlabel("Número de participantes")
    plt.ylabel("Resposta")

    for barra, n, pct in zip(
        barras,
        frequencia.values,
        percentual.values
    ):
        plt.text(
            barra.get_width() + 0.2,
            barra.get_y() + barra.get_height() / 2,
            f"{n} ({pct:.1f}%)",
            va="center"
        )

    plt.xlim(0, max(frequencia.values) + 8)

    salvar_grafico(f"q{numero}_distribuicao.png")

    return tabela


def gerar_cruzamento(nome, grupo, resposta):
    """
    Gera:
    - tabela de frequências absolutas;
    - tabela de percentuais por linha;
    - gráfico de barras empilhadas em 100%.
    """

    tabela_n = pd.crosstab(
        grupo,
        resposta,
        dropna=False
    )

    ordem = ordenar_respostas(resposta)

    tabela_n = tabela_n.reindex(columns=ordem, fill_value=0)

    tabela_pct = tabela_n.div(
        tabela_n.sum(axis=1),
        axis=0
    ) * 100

    tabela_n.to_csv(
        PASTA_SAIDA / f"{nome}_frequencias.csv",
        encoding="utf-8-sig"
    )

    tabela_pct.to_csv(
        PASTA_SAIDA / f"{nome}_percentuais.csv",
        encoding="utf-8-sig"
    )

    print(f"\nCRUZAMENTO {nome.upper()} — FREQUÊNCIAS")
    print(tabela_n.to_string())

    print(f"\nCRUZAMENTO {nome.upper()} — PERCENTUAIS POR GRUPO")
    print(tabela_pct.round(2).to_string())

    ax = tabela_pct.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        colormap="viridis"
    )

    ax.set_title(nome.replace("_", " ").upper())
    ax.set_xlabel("Padrão de utilização sem revisão detalhada")
    ax.set_ylabel("Percentual dentro de cada grupo (%)")
    ax.set_ylim(0, 100)

    ax.legend(
        title="Resposta",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.tick_params(axis="x", rotation=0)

    salvar_grafico(f"{nome}.png")

    return tabela_n, tabela_pct


# ============================================================
# LEITURA DO CSV
# ============================================================

if CAMINHO_CSV is None:
    raise FileNotFoundError(
        "CSV não encontrado. Verifique o caminho em POSSIVEIS_ARQUIVOS."
    )

df = pd.read_csv(
    CAMINHO_CSV,
    encoding="utf-8-sig"
)

print("=" * 70)
print("ANÁLISE DA SEÇÃO 5.4 — AUTONOMIA E COMPREENSÃO DE CÓDIGO")
print("=" * 70)

print(f"\nArquivo: {CAMINHO_CSV}")
print(f"Participantes: {len(df)}")


# ============================================================
# LOCALIZAÇÃO DAS QUESTÕES
# ============================================================

colunas = {
    numero: localizar_coluna(numero)
    for numero in [10, 11, 12, 13, 19]
}

print("\nQUESTÕES IDENTIFICADAS:")

for numero, coluna in colunas.items():
    print(f"Q{numero}: {coluna}")


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

dados = pd.DataFrame()

for numero, coluna in colunas.items():
    dados[f"Q{numero}"] = (
        df[coluna]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

print("\nRESPOSTAS AUSENTES:")

for numero in colunas:
    print(
        f"Q{numero}: "
        f"{dados[f'Q{numero}'].isna().sum()}"
    )


# ============================================================
# ETAPA 1 — DISTRIBUIÇÕES GERAIS
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 1 — DISTRIBUIÇÕES GERAIS")
print("=" * 70)

for numero in [10, 11, 12, 13, 19]:
    gerar_distribuicao(
        numero,
        dados[f"Q{numero}"].dropna()
    )


# ============================================================
# ETAPA 2 — AGRUPAMENTO DA Q11
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 2 — AGRUPAMENTO DA Q11")
print("=" * 70)

grupo_q11 = {
    "Nunca.": "Nunca/Raramente",
    "Raramente.": "Nunca/Raramente",
    "Às vezes.": "Às vezes/Frequentemente/Sempre",
    "Frequentemente.": "Às vezes/Frequentemente/Sempre",
    "Sempre.": "Às vezes/Frequentemente/Sempre"
}

dados["Grupo_Q11"] = dados["Q11"].map(grupo_q11)

print(
    dados["Grupo_Q11"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# ETAPA 3 — Q11 × Q12
# Revisão insuficiente × dificuldade de manutenção
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 3 — Q11 × Q12")
print("=" * 70)

gerar_cruzamento(
    "q11_q12_revisao_manutencao",
    dados["Grupo_Q11"],
    dados["Q12"]
)


# ============================================================
# ETAPA 4 — Q11 × Q13
# Revisão insuficiente × capacidade de explicar o código
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 4 — Q11 × Q13")
print("=" * 70)

gerar_cruzamento(
    "q11_q13_revisao_explicacao",
    dados["Grupo_Q11"],
    dados["Q13"]
)


# ============================================================
# ETAPA 5 — Q11 × Q19
# Revisão insuficiente × dificuldade sem IA
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 5 — Q11 × Q19")
print("=" * 70)

gerar_cruzamento(
    "q11_q19_revisao_autonomia",
    dados["Grupo_Q11"],
    dados["Q19"]
)


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA")
print("=" * 70)

print("\nArquivos gerados:")

for arquivo in sorted(PASTA_SAIDA.iterdir()):
    if arquivo.is_file():
        print(f"- {arquivo.name}")

print("\nObservações:")
print("- Os cruzamentos apresentam percentuais por linha.")
print("- As respostas ausentes são excluídas de cada cruzamento.")
print("- Q11 foi agrupada em dois níveis de frequência.")
print("- Os resultados são descritivos e não demonstram causalidade.")
