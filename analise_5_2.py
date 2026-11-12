from pathlib import Path
import re
import sys
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt

CAMINHO_CSV = Path("dados/data.csv")
PASTA_RESULTADOS = Path("resultados/analise_5_2")

PREFIXOS_COLUNAS = {
    "Q7": "7)",
    "Q8": "8)",
    "Q9": "9)",
}

NOMES = {
    "Q7": "Frequência de utilização de ferramentas de IAG",
    "Q8": "Ferramentas de IAG utilizadas",
    "Q9": "Finalidades de utilização da IAG",
}

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

def normalizar_texto(valor):
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto)
    return texto

def localizar_coluna(df, prefixo):
    candidatas = [
        coluna for coluna in df.columns
        if normalizar_texto(coluna).startswith(normalizar_texto(prefixo))
    ]

    if len(candidatas) == 1:
        return candidatas[0]

    if len(candidatas) > 1:
        print(f"\nERRO: mais de uma coluna encontrada para {prefixo}:")
        for coluna in candidatas:
            print(f"- {coluna}")
        sys.exit(1)

    print(f"\nERRO: não encontrei a coluna da questão {prefixo}")
    print("\nColunas disponíveis no CSV:")
    for coluna in df.columns:
        print(f"- {coluna}")
    sys.exit(1)


def preparar_respostas(serie):
    """Remove espaços externos e transforma strings vazias em ausentes."""
    serie = serie.astype("string").str.strip()
    return serie.replace("", pd.NA)


def tabela_frequencia_unica(serie):
    """Frequência absoluta e percentual para questão de resposta única."""
    respostas = preparar_respostas(serie).dropna()
    frequencias = respostas.value_counts()

    n_validos = len(respostas)
    percentuais = (
        frequencias / n_validos * 100
        if n_validos
        else frequencias.astype(float)
    )

    tabela = pd.DataFrame({
        "Frequência": frequencias,
        "Percentual (%)": percentuais.round(2),
    })

    return tabela, n_validos


def separar_multiplas_respostas(valor, alternativas_conhecidas):
    """
    Separa alternativas usando ponto e vírgula ou vírgula antes de uma
    alternativa conhecida. Isso evita dividir incorretamente respostas
    personalizadas de 'Outro' que contenham vírgulas.
    """
    if pd.isna(valor):
        return []

    texto = str(valor).strip()
    texto = re.sub(r"\s*;\s*", "|", texto)

    alternativas = sorted(alternativas_conhecidas, key=len, reverse=True)
    padrao = r",\s*(?=(?:" + "|".join(re.escape(x) for x in alternativas) + r")(?:\b|\s|\.|$))"
    texto = re.sub(padrao, "|", texto, flags=re.IGNORECASE)

    return [parte.strip() for parte in texto.split("|") if parte.strip()]


def tabela_frequencia_multipla(serie, alternativas_conhecidas):
    """
    Conta participantes por alternativa.
    Cada alternativa é contada no máximo uma vez por participante.
    """
    respostas = preparar_respostas(serie)
    respostas_validas = respostas.dropna()

    contagens = {}

    for resposta in respostas_validas:
        alternativas = set(
            separar_multiplas_respostas(resposta, alternativas_conhecidas)
        )
        for alternativa in alternativas:
            contagens[alternativa] = contagens.get(alternativa, 0) + 1

    n_validos = len(respostas_validas)

    tabela = pd.DataFrame(
        [{"Alternativa": nome, "Participantes": quantidade}
         for nome, quantidade in contagens.items()]
    )

    if not tabela.empty:
        tabela["Percentual dos participantes (%)"] = (
            tabela["Participantes"] / n_validos * 100
        ).round(2)
        tabela = tabela.sort_values(
            ["Participantes", "Alternativa"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return tabela, n_validos


def salvar_grafico_unico(tabela, titulo, nome_arquivo):
    """Gera gráfico de barras para questão de resposta única."""
    if tabela.empty:
        return

    tabela_plot = tabela.sort_values("Frequência", ascending=True)

    altura = max(4.5, len(tabela_plot) * 0.65)
    plt.figure(figsize=(9, altura))

    barras = plt.barh(
        tabela_plot["Categoria"].astype(str),
        tabela_plot["Frequência"]
    )

    plt.xlabel("Quantidade de participantes")
    plt.title(titulo)
    plt.grid(axis="x", alpha=0.25)

    total = int(tabela_plot["Frequência"].sum())
    for barra in barras:
        valor = int(barra.get_width())
        percentual = valor / total * 100 if total else 0
        plt.text(
            valor + 0.15,
            barra.get_y() + barra.get_height() / 2,
            f"{valor} ({percentual:.1f}%)",
            va="center"
        )

    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close()


def salvar_grafico_multipla(tabela, titulo, nome_arquivo):
    """Gera gráfico horizontal para questão de múltipla seleção."""
    if tabela.empty:
        return

    tabela_plot = tabela.sort_values("Participantes", ascending=True)
    altura = max(5, len(tabela_plot) * 0.48)

    plt.figure(figsize=(10, altura))
    barras = plt.barh(
        tabela_plot["Alternativa"],
        tabela_plot["Participantes"]
    )

    plt.xlabel("Quantidade de participantes")
    plt.title(titulo)
    plt.grid(axis="x", alpha=0.25)

    for barra, (_, linha) in zip(barras, tabela_plot.iterrows()):
        quantidade = int(linha["Participantes"])
        percentual = linha["Percentual dos participantes (%)"]
        plt.text(
            quantidade + 0.15,
            barra.get_y() + barra.get_height() / 2,
            f"{quantidade} ({percentual:.1f}%)",
            va="center"
        )

    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# CARREGAMENTO E VALIDAÇÃO
# ---------------------------------------------------------------------

print("\n" + "=" * 76)
print("ANÁLISE DA SEÇÃO 5.2 — PERFIL DE ADOÇÃO E UTILIZAÇÃO DA IAG")
print("=" * 76)

if not CAMINHO_CSV.exists():
    sys.exit(
        f"\nERRO: arquivo não encontrado: {CAMINHO_CSV}\n"
        "Execute o script a partir da raiz do projeto TCC."
    )

df = pd.read_csv(CAMINHO_CSV)

print(f"\nTotal de participantes na base: {len(df)}")

colunas = {
    codigo: localizar_coluna(df, prefixo)
    for codigo, prefixo in PREFIXOS_COLUNAS.items()
}

for codigo, coluna in colunas.items():
    print(f"{codigo}: {coluna}")


# ---------------------------------------------------------------------
# Q7 — FREQUÊNCIA DE UTILIZAÇÃO
# ---------------------------------------------------------------------

print("\n" + "-" * 76)
print("Q7 — FREQUÊNCIA DE UTILIZAÇÃO DE FERRAMENTAS DE IAG")
print("-" * 76)

tabela_q7, n_q7 = tabela_frequencia_unica(df[colunas["Q7"]])
ausentes_q7 = len(df) - n_q7

print(f"Respostas válidas: {n_q7}")
print(f"Respostas ausentes: {ausentes_q7}")
print(tabela_q7.to_string())

tabela_q7_export = tabela_q7.rename_axis("Categoria").reset_index()
tabela_q7_export.to_csv(
    PASTA_RESULTADOS / "q7_frequencias.csv",
    index=False,
    encoding="utf-8-sig"
)

salvar_grafico_unico(
    tabela_q7_export,
    "Q7 — Frequência de utilização de ferramentas de IAG",
    "q7_frequencia_utilizacao.png"
)


# ---------------------------------------------------------------------
# Q8 — FERRAMENTAS UTILIZADAS (MÚLTIPLA SELEÇÃO)
# ---------------------------------------------------------------------

print("\n" + "-" * 76)
print("Q8 — FERRAMENTAS DE IAG UTILIZADAS")
print("-" * 76)

OPCOES_Q8 = ["Claude", "ChatGPT", "GitHub Copilot", "Antigravity", "Cursor"]
tabela_q8, n_q8 = tabela_frequencia_multipla(
    df[colunas["Q8"]], OPCOES_Q8
)
ausentes_q8 = len(df) - n_q8
total_selecoes_q8 = int(tabela_q8["Participantes"].sum()) if not tabela_q8.empty else 0

print(f"Participantes com resposta válida: {n_q8}")
print(f"Respostas ausentes: {ausentes_q8}")
print(f"Total de seleções contabilizadas: {total_selecoes_q8}")
print(
    f"Média de ferramentas selecionadas por participante: "
    f"{total_selecoes_q8 / n_q8:.2f}" if n_q8 else
    "Média de ferramentas selecionadas por participante: não calculável"
)
print(tabela_q8.to_string(index=False) if not tabela_q8.empty else "Sem respostas válidas.")

tabela_q8.to_csv(
    PASTA_RESULTADOS / "q8_ferramentas_frequencias.csv",
    index=False,
    encoding="utf-8-sig"
)

salvar_grafico_multipla(
    tabela_q8,
    "Q8 — Ferramentas de IAG utilizadas pelos participantes",
    "q8_ferramentas_utilizadas.png"
)


# ---------------------------------------------------------------------
# Q9 — FINALIDADES DE UTILIZAÇÃO (MÚLTIPLA SELEÇÃO)
# ---------------------------------------------------------------------

print("\n" + "-" * 76)
print("Q9 — FINALIDADES DE UTILIZAÇÃO DA IAG")
print("-" * 76)

OPCOES_Q9 = [
    "Codificação", "Debugging", "Pesquisa", "Criação de conteúdo",
    "Revisão de código", "Aprendizado de conceitos novos"
]
tabela_q9, n_q9 = tabela_frequencia_multipla(
    df[colunas["Q9"]], OPCOES_Q9
)
ausentes_q9 = len(df) - n_q9
total_selecoes_q9 = int(tabela_q9["Participantes"].sum()) if not tabela_q9.empty else 0

print(f"Participantes com resposta válida: {n_q9}")
print(f"Respostas ausentes: {ausentes_q9}")
print(f"Total de seleções contabilizadas: {total_selecoes_q9}")
print(
    f"Média de finalidades selecionadas por participante: "
    f"{total_selecoes_q9 / n_q9:.2f}" if n_q9 else
    "Média de finalidades selecionadas por participante: não calculável"
)
print(tabela_q9.to_string(index=False) if not tabela_q9.empty else "Sem respostas válidas.")

tabela_q9.to_csv(
    PASTA_RESULTADOS / "q9_finalidades_frequencias.csv",
    index=False,
    encoding="utf-8-sig"
)

salvar_grafico_multipla(
    tabela_q9,
    "Q9 — Finalidades de utilização da IAG",
    "q9_finalidades_utilizacao.png"
)


# ---------------------------------------------------------------------
# RESUMO E ARQUIVOS GERADOS
# ---------------------------------------------------------------------

print("\n" + "=" * 76)
print("RESUMO DA ANÁLISE 5.2")
print("=" * 76)
print(f"Participantes na base: {len(df)}")
print(f"Q7 — respostas válidas: {n_q7}; ausentes: {ausentes_q7}")
print(f"Q8 — respostas válidas: {n_q8}; ausentes: {ausentes_q8}")
print(f"Q9 — respostas válidas: {n_q9}; ausentes: {ausentes_q9}")

print("\nArquivos gerados em:")
print(PASTA_RESULTADOS.resolve())

print("\nTabelas:")
for nome in [
    "q7_frequencias.csv",
    "q8_ferramentas_frequencias.csv",
    "q9_finalidades_frequencias.csv",
]:
    print(f"- {nome}")

print("\nGráficos:")
for nome in [
    "q7_frequencia_utilizacao.png",
    "q8_ferramentas_utilizadas.png",
    "q9_finalidades_utilizacao.png",
]:
    print(f"- {nome}")

print("\nAnálise 5.2 concluída.")
