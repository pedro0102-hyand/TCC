from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

candidatos = []
for pasta in (BASE_DIR / "dados", BASE_DIR / "data", BASE_DIR):
    candidatos.extend([
        pasta / "data.csv",
        pasta / "respostas_tcc.csv",
    ])

ARQUIVO = next((p for p in candidatos if p.exists()), None)

if ARQUIVO is None:
    raise FileNotFoundError(
        "Não encontrei a base CSV. Coloque data.csv ou respostas_tcc.csv "
        "na pasta dados/, data/ ou na mesma pasta deste script."
    )

try:
    df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(ARQUIVO, encoding="latin-1")

SAIDA = BASE_DIR / "resultados" / "analise_5_3"
SAIDA.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("ANÁLISE DA HIPÓTESE H1 — EXPERIÊNCIA TÉCNICA E PADRÕES DE INTERAÇÃO")
print("=" * 78)
print(f"Arquivo: {ARQUIVO}")
print(f"Participantes na base: {len(df)}")


# ---------------------------------------------------------------------
# 2. Identificação das colunas e limpeza básica
# ---------------------------------------------------------------------

def coluna_questao(numero):
    """Localiza a coluna pelo número inicial da questão."""
    prefixo = f"{numero})"
    encontradas = [c for c in df.columns if str(c).strip().startswith(prefixo)]
    if not encontradas:
        raise KeyError(f"Não encontrei a coluna da questão {numero}.")
    return encontradas[0]


questoes = {
    "Q2_Escolaridade": 2,
    "Q3_Conhecimento": 3,
    "Q4_Frequencia_Codigo": 4,
    "Q5_Tempo_Programacao": 5,
    "Q7_Frequencia_IA": 7,
    "Q14_Aceita_Sem_Compreender": 14,
    "Q15_Tenta_Sozinho": 15,
    "Q18_Reacao_IA_Falha": 18,
    "Q25_Padrao_Interacao": 25,
}

dados = pd.DataFrame(index=df.index)
for nome, numero in questoes.items():
    col = coluna_questao(numero)
    dados[nome] = df[col].fillna("").astype(str).str.strip()
    dados.loc[dados[nome].isin(["", "nan", "NaN"]), nome] = pd.NA

print("\nQuestões identificadas:")
for nome, numero in questoes.items():
    print(f"  {nome}: {coluna_questao(numero)}")


# ---------------------------------------------------------------------
# 3. Distribuições individuais das variáveis
# ---------------------------------------------------------------------

variaveis_perfil = [
    "Q2_Escolaridade",
    "Q3_Conhecimento",
    "Q4_Frequencia_Codigo",
    "Q5_Tempo_Programacao",
]

variaveis_interacao = [
    "Q7_Frequencia_IA",
    "Q14_Aceita_Sem_Compreender",
    "Q15_Tenta_Sozinho",
    "Q25_Padrao_Interacao",
]

linhas_distribuicao = []

print("\n" + "=" * 78)
print("DISTRIBUIÇÕES DAS VARIÁVEIS")
print("=" * 78)

for variavel in variaveis_perfil + variaveis_interacao:
    serie = dados[variavel].dropna()
    contagem = serie.value_counts()
    percentual = serie.value_counts(normalize=True) * 100

    print(f"\n{variavel} — válidas: {len(serie)}")
    for categoria, n in contagem.items():
        perc = percentual[categoria]
        print(f"  {categoria}: {n} ({perc:.2f}%)")
        linhas_distribuicao.append({
            "variavel": variavel,
            "categoria": categoria,
            "n": int(n),
            "percentual": round(float(perc), 2),
            "respostas_validas": int(len(serie)),
        })

pd.DataFrame(linhas_distribuicao).to_csv(
    SAIDA / "distribuicoes.csv", index=False, encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 4. Cruzamentos entre perfil técnico e padrões de interação
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("CRUZAMENTOS — CONTAGENS E PERCENTUAIS POR LINHA")
print("Os percentuais mostram a distribuição da variável de interação")
print("dentro de cada categoria do perfil técnico.")
print("=" * 78)

linhas_contagens = []
linhas_percentuais = []

for perfil in variaveis_perfil:
    for interacao in variaveis_interacao:
        tabela = pd.crosstab(
            dados[perfil],
            dados[interacao],
            dropna=False
        )

        tabela_pct = pd.crosstab(
            dados[perfil],
            dados[interacao],
            normalize="index"
        ) * 100

        print(f"\n\n{perfil} × {interacao}")
        print("\nContagens:")
        print(tabela.to_string())
        print("\nPercentuais por linha:")
        print(tabela_pct.round(1).to_string())

        for categoria_perfil in tabela.index:
            for categoria_interacao in tabela.columns:
                n = int(tabela.loc[categoria_perfil, categoria_interacao])
                pct = float(tabela_pct.loc[categoria_perfil, categoria_interacao])
                linhas_contagens.append({
                    "variavel_perfil": perfil,
                    "categoria_perfil": categoria_perfil,
                    "variavel_interacao": interacao,
                    "categoria_interacao": categoria_interacao,
                    "n": n,
                })
                linhas_percentuais.append({
                    "variavel_perfil": perfil,
                    "categoria_perfil": categoria_perfil,
                    "variavel_interacao": interacao,
                    "categoria_interacao": categoria_interacao,
                    "percentual_linha": round(pct, 2),
                })

pd.DataFrame(linhas_contagens).to_csv(
    SAIDA / "cruzamentos_contagens.csv", index=False, encoding="utf-8-sig"
)
pd.DataFrame(linhas_percentuais).to_csv(
    SAIDA / "cruzamentos_percentuais_linha.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 5. Questão Q18 — respostas de múltipla seleção
# ---------------------------------------------------------------------

q18_col = coluna_questao(18)
respostas_q18 = df[q18_col].fillna("").astype(str).str.strip()

# As opções do formulário são separadas por ponto e vírgula.
# O ponto e vírgula é usado para preservar vírgulas que possam existir
# dentro de uma resposta personalizada.
selecoes_q18 = []

for resposta in respostas_q18:
    if not resposta or resposta.lower() == "nan":
        continue
    for opcao in resposta.split(";"):
        opcao = opcao.strip()
        if opcao:
            selecoes_q18.append(opcao)

contagem_q18 = pd.Series(selecoes_q18).value_counts()
total_selecoes_q18 = int(contagem_q18.sum())
validas_q18 = int(respostas_q18.replace("", pd.NA).notna().sum())

print("\n" + "=" * 78)
print("Q18 — COMPORTAMENTOS QUANDO A IA NÃO RESOLVE O PROBLEMA")
print("=" * 78)
print(f"Respostas válidas: {validas_q18}")
print(f"Total de seleções: {total_selecoes_q18}")
print("\nFrequência por comportamento:")

q18_resultados = []
for opcao, n in contagem_q18.items():
    percentual = (n / validas_q18 * 100) if validas_q18 else 0
    print(f"  {opcao}: {n} ({percentual:.2f}% dos participantes)")
    q18_resultados.append({
        "comportamento": opcao,
        "n": int(n),
        "percentual_participantes": round(float(percentual), 2),
        "respostas_validas": validas_q18,
    })

pd.DataFrame(q18_resultados).to_csv(
    SAIDA / "q18_comportamentos.csv", index=False, encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 6. Resumo e alertas de interpretação
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("ARQUIVOS GERADOS")
print("=" * 78)
for arquivo in sorted(SAIDA.glob("*.csv")):
    print(f"  {arquivo}")

print("\nOBSERVAÇÕES METODOLÓGICAS:")
print("- Q18 permite múltiplas seleções; seus percentuais podem ultrapassar 100%.")
print("- Q25 contém respostas personalizadas; elas foram mantidas separadas.")
print("- Categorias com poucos participantes devem ser interpretadas com cautela.")
print("- Os cruzamentos são descritivos e não demonstram causalidade.")
print("- Q2 (escolaridade) não deve ser confundida com maturidade técnica.")
print("- Q5 combina tempo de estudo e/ou trabalho com programação.")
print("\nAnálise concluída.")
