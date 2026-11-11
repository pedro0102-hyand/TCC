from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CAMINHO_CSV = BASE_DIR / "dados" / "data.csv"
PASTA_RESULTADOS = BASE_DIR / "resultados" / "analise_5_1"

COLUNAS = {
    "Q1": "1) Qual é a sua principal relação com a área de Tecnologia da Informação?",
    "Q2": "2) Qual o seu grau de escolaridade?",
    "Q3": "3) Como você avalia seu nível de conhecimento em programação?",
    "Q4": "4) Com que frequência você desenvolve software ou escreve código?",
    "Q5": "5) Há quanto tempo você estuda e/ou trabalha com programação?",
    "Q6": "6) Qual sua área/função principal ?",
}

NOMES = {
    "Q1": "Relação com TI",
    "Q2": "Escolaridade",
    "Q3": "Conhecimento em programação",
    "Q4": "Frequência de desenvolvimento",
    "Q5": "Tempo de estudo e/ou trabalho com programação",
    "Q6": "Área ou função principal",
}

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

def preparar_resposta_unica(serie):
    serie = serie.astype("string").str.strip()
    return serie.replace("", pd.NA)

def tabela_frequencia(serie):
    respostas = preparar_resposta_unica(serie).dropna()
    frequencias = respostas.value_counts()
    percentuais = (frequencias / len(respostas) * 100) if len(respostas) else frequencias
    return pd.DataFrame({"Frequência": frequencias, "Percentual (%)": percentuais.round(2)}), len(respostas)

def salvar_grafico_frequencia(tabela, titulo, nome_arquivo, horizontal=False):
    if tabela.empty:
        return
    if horizontal:
        altura = max(5, len(tabela) * 0.42)
        plt.figure(figsize=(11, altura))
        barras = plt.barh(tabela.index.astype(str), tabela["Frequência"])
        plt.xlabel("Quantidade de participantes")
        plt.ylabel("")
        for barra in barras:
            valor = barra.get_width()
            plt.text(valor + 0.15, barra.get_y() + barra.get_height() / 2, f"{int(valor)}", va="center")
    else:
        largura = max(8, len(tabela) * 1.25)
        plt.figure(figsize=(largura, 5.5))
        barras = plt.bar(tabela.index.astype(str), tabela["Frequência"])
        plt.ylabel("Quantidade de participantes")
        plt.xlabel("")
        plt.xticks(rotation=25, ha="right")
        for barra in barras:
            valor = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, valor + 0.15, f"{int(valor)}", ha="center", va="bottom")
    plt.title(titulo)
    plt.grid(axis="y" if not horizontal else "x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close()

print("\n" + "=" * 72)
print("ANÁLISE CONSOLIDADA — SEÇÃO 5.1")
print("CARACTERIZAÇÃO DOS PARTICIPANTES")
print("=" * 72)

if not CAMINHO_CSV.exists():
    sys.exit(f"\nERRO: arquivo não encontrado: {CAMINHO_CSV}")

df = pd.read_csv(CAMINHO_CSV)

colunas_ausentes = [col for col in COLUNAS.values() if col not in df.columns]
if colunas_ausentes:
    print("\nERRO: não encontrei estas colunas no CSV:")
    for coluna in colunas_ausentes:
        print(f"- {coluna}")
    print("\nColunas disponíveis:")
    for coluna in df.columns:
        print(f"- {coluna}")
    sys.exit(1)

print(f"\nTotal de participantes na base: {len(df)}")

tabelas = {}
print("\n" + "-" * 72)
print("DISTRIBUIÇÕES Q1 A Q5")
print("-" * 72)

for codigo in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    coluna = COLUNAS[codigo]
    tabela, n_validos = tabela_frequencia(df[coluna])
    tabelas[codigo] = tabela
    print(f"\n{codigo} — {NOMES[codigo]}")
    print(f"Respostas válidas: {n_validos}")
    print(f"Respostas ausentes: {len(df) - n_validos}")
    print(tabela.to_string())
    tabela.to_csv(PASTA_RESULTADOS / f"{codigo.lower()}_frequencias.csv", encoding="utf-8-sig")
    salvar_grafico_frequencia(tabela, f"{codigo} — {NOMES[codigo]}", f"{codigo.lower()}_distribuicao.png")

print("\n" + "-" * 72)
print("Q6 — ÁREAS E FUNÇÕES (MÚLTIPLA SELEÇÃO)")
print("-" * 72)

q6 = preparar_resposta_unica(df[COLUNAS["Q6"]])
selecoes_por_participante = []

for resposta in q6.dropna():
    areas = [area.strip() for area in str(resposta).split(";") if area.strip()]
    areas = list(dict.fromkeys(areas))
    if areas:
        selecoes_por_participante.append(areas)

total_q6_validos = len(selecoes_por_participante)
todas_areas = [area for areas in selecoes_por_participante for area in areas]

if total_q6_validos:
    freq_q6 = pd.Series(todas_areas).value_counts()
    perc_q6 = freq_q6 / total_q6_validos * 100
    tabela_q6 = pd.DataFrame({"Participantes": freq_q6, "Percentual (%)": perc_q6.round(2)})
else:
    tabela_q6 = pd.DataFrame(columns=["Participantes", "Percentual (%)"])

total_selecoes = len(todas_areas)
media_selecoes = total_selecoes / total_q6_validos if total_q6_validos else 0

print(f"Participantes na base: {len(df)}")
print(f"Participantes com resposta válida: {total_q6_validos}")
print(f"Respostas ausentes ou sem seleção: {len(df) - total_q6_validos}")
print(f"Total de seleções: {total_selecoes}")
print(f"Média de áreas selecionadas por participante: {media_selecoes:.2f}")
print("\nDistribuição por área:")
if not tabela_q6.empty:
    print(tabela_q6.to_string())
else:
    print("Nenhuma seleção válida encontrada.")

print("\nNota: percentuais calculados sobre participantes com resposta válida;")
print("podem somar mais de 100% porque a questão permite múltiplas seleções.")

tabela_q6.to_csv(PASTA_RESULTADOS / "q6_areas_frequencias.csv", encoding="utf-8-sig")
salvar_grafico_frequencia(tabela_q6.rename(columns={"Participantes": "Frequência"}), "Q6 — Participantes por área ou função", "q6_areas.png", horizontal=True)

print("\n" + "-" * 72)
print("CRUZAMENTO Q3 X Q5 — CONHECIMENTO X TEMPO DE EXPERIÊNCIA")
print("-" * 72)

dados_cruzamento = df[[COLUNAS["Q3"], COLUNAS["Q5"]]].copy()

for coluna in [COLUNAS["Q3"], COLUNAS["Q5"]]:
    dados_cruzamento[coluna] = preparar_resposta_unica(dados_cruzamento[coluna])

dados_cruzamento = dados_cruzamento.dropna()
frequencias_cruzadas = pd.crosstab(dados_cruzamento[COLUNAS["Q5"]], dados_cruzamento[COLUNAS["Q3"]])
percentuais_cruzados = (pd.crosstab(dados_cruzamento[COLUNAS["Q5"]], dados_cruzamento[COLUNAS["Q3"]], normalize="index") * 100).round(2)

print(f"Participantes considerados: {len(dados_cruzamento)}")
print("\nFrequências absolutas:")
print(frequencias_cruzadas.to_string())
print("\nPercentuais dentro de cada faixa de experiência:")
print(percentuais_cruzados.to_string())
print("\nTotal por faixa de experiência:")
print(frequencias_cruzadas.sum(axis=1).to_string())
print("\nTotal por nível de conhecimento:")
print(frequencias_cruzadas.sum(axis=0).to_string())

frequencias_cruzadas.to_csv(PASTA_RESULTADOS / "cruzamento_q3_q5_frequencias.csv", encoding="utf-8-sig")
percentuais_cruzados.to_csv(PASTA_RESULTADOS / "cruzamento_q3_q5_percentuais.csv", encoding="utf-8-sig")

if not percentuais_cruzados.empty:
    ax = percentuais_cruzados.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title("Q3 x Q5 — Conhecimento por tempo de experiência")
    ax.set_xlabel("Tempo de estudo e/ou trabalho com programação")
    ax.set_ylabel("Percentual dentro da faixa de experiência (%)")
    ax.legend(title="Conhecimento", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "cruzamento_q3_q5_percentuais.png", dpi=300, bbox_inches="tight")
    plt.close()

resumo = []

for codigo in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    tabela = tabelas[codigo]
    if not tabela.empty:
        for categoria, linha in tabela.iterrows():
            resumo.append({"Questão": codigo, "Variável": NOMES[codigo], "Categoria": categoria, "Frequência": int(linha["Frequência"]), "Percentual (%)": float(linha["Percentual (%)"])})

for categoria, linha in tabela_q6.iterrows():
    resumo.append({"Questão": "Q6", "Variável": NOMES["Q6"], "Categoria": categoria, "Frequência": int(linha["Participantes"]), "Percentual (%)": float(linha["Percentual (%)"])})
pd.DataFrame(resumo).to_csv(PASTA_RESULTADOS / "resumo_consolidado_5_1.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 72)
print("ANÁLISE 5.1 FINALIZADA")
print("=" * 72)
print(f"\nArquivos gerados em: {PASTA_RESULTADOS}/")
print("- CSVs com frequências de Q1 a Q6")
print("- CSVs do cruzamento Q3 x Q5")
print("- resumo_consolidado_5_1.csv")
print("- gráficos individuais de Q1 a Q6")
print("- gráfico do cruzamento Q3 x Q5")
