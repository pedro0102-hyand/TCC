
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import unicodedata

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "dados/data.csv"
PASTA_SAIDA = "resultados/analise_5_3"

os.makedirs(PASTA_SAIDA, exist_ok=True)

ESTRATEGIAS = {
    "Consulta outras fontes de estudo":
        "Consulto outras fontes de estudo.",

    "Tenta resolver sozinho utilizando conhecimentos":
        "Tento resolver sozinho utilizando meus conhecimentos.",

    "Reformula prompt":
        "Reformulo o prompt e tento novamente.",

    "Procura documentação oficial":
        "Procuro documentação oficial.",

    "Tem dificuldade para continuar sem IA":
        "Tenho dificuldade para continuar sem a IA.",

    "Pergunta a amigos/colegas":
        "Pergunto a amigos e colegas de trabalho.",
}


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

    # Remove pontuação e espaços extras
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto



def localizar_coluna(df, numero):
    """
    Localiza a coluna pelo número da questão,
    independentemente da pontuação do cabeçalho.
    """

    for coluna in df.columns:

        nome = normalizar(coluna)

        # Aceita: 3), 3., 3 -, Q3, Questão 3
        padrao = rf"^(?:q\s*|questao\s*)?{numero}(?:\s|$)"

        if re.match(padrao, nome):
            return coluna

    raise ValueError(
        f"Não foi possível localizar a coluna da Q{numero}.\n"
        f"Colunas disponíveis:\n{list(df.columns)}"
    )



def separar_respostas(valor):
    """
    Separa respostas de múltipla seleção.
    Considera ponto e vírgula ou vírgula como separadores.
    """

    if pd.isna(valor):
        return []

    return [
        item.strip()
        for item in re.split(r"[;,]", str(valor))
        if item.strip()
    ]


# ============================================================
# LEITURA DA BASE
# ============================================================

df = pd.read_csv(ARQUIVO)

col_q3 = localizar_coluna(df, 3)
col_q5 = localizar_coluna(df, 5)
col_q18 = localizar_coluna(df, 18)

print("=" * 70)
print("ANÁLISE COMPLEMENTAR DA Q18 — HIPÓTESE H1")
print("=" * 70)

print(f"\nTotal de registros: {len(df)}")
print(f"Coluna Q3: {col_q3}")
print(f"Coluna Q5: {col_q5}")
print(f"Coluna Q18: {col_q18}")


# ============================================================
# CRIAÇÃO DOS INDICADORES DA Q18
# ============================================================

respostas_q18 = df[col_q18].apply(separar_respostas)

for nome, alternativa in ESTRATEGIAS.items():

    alternativa_norm = normalizar(alternativa)

    df[nome] = respostas_q18.apply(
        lambda respostas: int(
            any(
                normalizar(resposta) == alternativa_norm
                for resposta in respostas
            )
        )
    )


# ============================================================
# VERIFICAÇÃO DE RESPOSTAS NÃO RECONHECIDAS
# ============================================================

alternativas_reconhecidas = {
    normalizar(alternativa)
    for alternativa in ESTRATEGIAS.values()
}

nao_reconhecidas = set()

for respostas in respostas_q18:
    for resposta in respostas:

        if normalizar(resposta) not in alternativas_reconhecidas:
            nao_reconhecidas.add(resposta)

print("\nRespostas da Q18 não reconhecidas:")

if nao_reconhecidas:
    for resposta in sorted(nao_reconhecidas):
        print(f"- {resposta}")
else:
    print("Nenhuma.")


# ============================================================
# FUNÇÃO PARA REALIZAR OS CRUZAMENTOS
# ============================================================

def analisar_cruzamento(coluna_perfil, nome_perfil, arquivo_prefixo):

    print("\n" + "=" * 70)
    print(f"Q18 × {nome_perfil}")
    print("=" * 70)

    base = df.dropna(subset=[coluna_perfil]).copy()

    resultados = []

    for grupo, participantes in base.groupby(coluna_perfil):

        n_grupo = len(participantes)

        print(f"\n{grupo} (n={n_grupo})")

        for estrategia in ESTRATEGIAS:

            quantidade = participantes[estrategia].sum()

            percentual = (
                quantidade / n_grupo * 100
                if n_grupo > 0 else 0
            )

            print(
                f"  {estrategia}: "
                f"{quantidade} ({percentual:.2f}%)"
            )

            resultados.append({
                "Grupo": grupo,
                "N do grupo": n_grupo,
                "Estratégia": estrategia,
                "Quantidade": quantidade,
                "Percentual": round(percentual, 2)
            })

    resultado_df = pd.DataFrame(resultados)

    # Exportação CSV
    caminho_csv = os.path.join(
        PASTA_SAIDA,
        f"{arquivo_prefixo}.csv"
    )

    resultado_df.to_csv(
        caminho_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    tabela = resultado_df.pivot(
        index="Grupo",
        columns="Estratégia",
        values="Percentual"
    )

    ax = tabela.plot(
        kind="bar",
        figsize=(13, 7),
        width=0.8
    )

    ax.set_title(
        f"Estratégias quando a IA não resolve — {nome_perfil}"
    )

    ax.set_xlabel(nome_perfil)

    ax.set_ylabel(
        "Participantes que selecionaram a estratégia (%)"
    )

    ax.set_ylim(0, 100)

    ax.legend(
        title="Estratégias",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    caminho_grafico = os.path.join(
        PASTA_SAIDA,
        f"{arquivo_prefixo}.png"
    )

    plt.savefig(
        caminho_grafico,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nCSV salvo em: {caminho_csv}")
    print(f"Gráfico salvo em: {caminho_grafico}")


# ============================================================
# EXECUÇÃO DOS CRUZAMENTOS
# ============================================================

analisar_cruzamento(
    col_q3,
    "Conhecimento em programação (Q3)",
    "q18_por_conhecimento_q3"
)

analisar_cruzamento(
    col_q5,
    "Tempo de experiência (Q5)",
    "q18_por_experiencia_q5"
)


# ============================================================
# RESUMO GERAL DA Q18
# ============================================================

print("\n" + "=" * 70)
print("RESUMO GERAL DA Q18")
print("=" * 70)

for estrategia in ESTRATEGIAS:

    quantidade = df[estrategia].sum()

    percentual = quantidade / len(df) * 100

    print(
        f"{estrategia}: "
        f"{quantidade} ({percentual:.2f}%)"
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISE COMPLEMENTAR CONCLUÍDA")
print("=" * 70)
