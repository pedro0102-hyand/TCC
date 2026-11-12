from pathlib import Path
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CAMINHO_CSV = BASE_DIR / "dados" / "data.csv"

PASTA_SAIDA = BASE_DIR / "resultados" / "analise_5_3"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 10


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar(texto):
    """Remove acentos, pontuação e diferenças de maiúsculas."""
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere for caractere in texto
        if not unicodedata.combining(caractere)
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def ordenar_categorias_presentes(serie, ordem):
    categorias = list(serie.dropna().unique())
    categorias_normalizadas = {
        normalizar(categoria): categoria
        for categoria in categorias
    }

    ordenadas = [
        categorias_normalizadas[normalizar(categoria)]
        for categoria in ordem
        if normalizar(categoria) in categorias_normalizadas
    ]
    normalizadas_ordenadas = {normalizar(categoria) for categoria in ordenadas}

    return ordenadas + [
        categoria for categoria in categorias
        if normalizar(categoria) not in normalizadas_ordenadas
    ]


def localizar_coluna(numero):
    """Localiza a coluna pelo número da questão."""
    prefixo = re.compile(rf"^\s*{numero}\s*[\).\-:]")

    encontradas = [
        coluna for coluna in df.columns
        if prefixo.search(str(coluna))
    ]

    if not encontradas:
        raise ValueError(
            f"Não foi possível localizar a coluna da questão {numero}."
        )

    if len(encontradas) > 1:
        print(
            f"Atenção: mais de uma coluna encontrada para Q{numero}. "
            f"Será utilizada: {encontradas[0]}"
        )

    return encontradas[0]


def salvar_grafico(nome):
    caminho = PASTA_SAIDA / nome
    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {caminho}")


def tabela_percentual_linha(dados, coluna_linha, coluna_coluna):
    """Calcula percentuais dentro de cada categoria da variável de linha."""
    tabela = pd.crosstab(
        dados[coluna_linha],
        dados[coluna_coluna],
        normalize="index"
    ) * 100

    return tabela


def grafico_barras_empilhadas(tabela, titulo, xlabel, nome_arquivo):
    ax = tabela.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        colormap="tab20"
    )

    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel(xlabel)
    ax.legend(
        title="Respostas",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax.tick_params(axis="x", rotation=20)

    salvar_grafico(nome_arquivo)


# ============================================================
# LEITURA DOS DADOS
# ============================================================

if not CAMINHO_CSV.exists():
    raise FileNotFoundError(
        f"Arquivo CSV não encontrado: {CAMINHO_CSV}"
    )

df = pd.read_csv(CAMINHO_CSV, encoding="utf-8-sig")

print("=" * 65)
print("ANÁLISE DOS GRÁFICOS — SEÇÃO 5.3")
print("=" * 65)
print(f"Participantes encontrados: {len(df)}")

col_q3 = localizar_coluna(3)
col_q5 = localizar_coluna(5)
col_q14 = localizar_coluna(14)
col_q18 = localizar_coluna(18)
col_q25 = localizar_coluna(25)

print("\nColunas identificadas:")
for numero, coluna in [
    (3, col_q3),
    (5, col_q5),
    (14, col_q14),
    (18, col_q18),
    (25, col_q25),
]:
    print(f"Q{numero}: {coluna}")


# ============================================================
# PREPARAÇÃO DAS VARIÁVEIS
# ============================================================

dados = df[[col_q3, col_q5, col_q14, col_q18, col_q25]].copy()

for coluna in [col_q3, col_q5, col_q14, col_q25]:
    dados[coluna] = dados[coluna].astype("string").str.strip()

dados = dados.dropna(subset=[col_q3, col_q5, col_q14, col_q25])

# Ordem desejada para as categorias de perfil
ordem_q3 = ordenar_categorias_presentes(dados[col_q3], [
    "Iniciante",
    "Básico",
    "Intermediário",
    "Avançado"
])

ordem_q5 = ordenar_categorias_presentes(dados[col_q5], [
    "Não tenho experiência",
    "Menos de 1 ano",
    "De 1 a 4 anos",
    "Mais de 4 anos"
])


# ============================================================
# GRÁFICO 1 — Q3 × Q14
# Conhecimento em programação × aceitação sem compreensão
# ============================================================

print("\n[1/4] Gerando Q3 × Q14...")

tabela_q3_q14 = tabela_percentual_linha(
    dados,
    col_q3,
    col_q14
)

tabela_q3_q14 = tabela_q3_q14.reindex(
    ordem_q3
).dropna(how="all")

tabela_q3_q14.to_csv(
    PASTA_SAIDA / "q3_q14_percentuais.csv",
    encoding="utf-8-sig"
)

grafico_barras_empilhadas(
    tabela_q3_q14,
    "Nível de conhecimento × frequência de aceitação de soluções sem compreensão",
    "Percentual dentro de cada nível de conhecimento (%)",
    "q3_q14_conhecimento_aceitacao.png"
)


# ============================================================
# GRÁFICO 2 — Q5 × Q25
# Tempo de experiência × padrão de interação
# ============================================================

print("\n[2/4] Gerando Q5 × Q25...")

# Reduz o tamanho dos rótulos para facilitar a leitura do gráfico.
def resumir_q25(valor):
    texto = normalizar(valor)

    if "planejo detalhadamente" in texto:
        return "Planeja e revisa a solução"

    if "estruturo a regra de negocio" in texto:
        return "Define regras e delega o código"

    if "comeco tentando resolver sozinho" in texto:
        return "Começa sozinho e depois delega"

    if "fac o perguntas" in texto or "faco perguntas" in texto:
        return "Questiona para compreender"

    if "procuro compreender o codigo" in texto:
        return "Compreende após receber a solução"

    if "tento resolver sozinho e uso a ia" in texto:
        return "Resolve sozinho e usa IA para dúvidas"

    # Mantém o texto original caso apareça uma resposta diferente.
    return str(valor)


dados["Q25_resumida"] = dados[col_q25].apply(resumir_q25)

tabela_q5_q25 = pd.crosstab(
    dados[col_q5],
    dados["Q25_resumida"],
    normalize="index"
) * 100

tabela_q5_q25 = tabela_q5_q25.reindex(ordem_q5).dropna(how="all")

tabela_q5_q25.to_csv(
    PASTA_SAIDA / "q5_q25_percentuais.csv",
    encoding="utf-8-sig"
)

grafico_barras_empilhadas(
    tabela_q5_q25,
    "Tempo de experiência × padrão de interação com a IA",
    "Percentual dentro de cada faixa de experiência (%)",
    "q5_q25_experiencia_interacao.png"
)


# ============================================================
# PREPARAÇÃO DA Q18 — MÚLTIPLA SELEÇÃO
# ============================================================

# Como Q18 permite múltiplas respostas, cada alternativa é
# contabilizada individualmente por participante.

opcoes_q18 = {
    "Consulta outras fontes": (
        "Consulto outras fontes de estudo."
    ),
    "Tenta resolver sozinho": (
        "Tento resolver sozinho utilizando meus conhecimentos."
    ),
    "Reformula o prompt": (
        "Reformulo o prompt e tento novamente."
    ),
    "Documentação oficial": (
        "Procuro documentação oficial."
    ),
    "Dificuldade sem IA": (
        "Tenho dificuldade para continuar sem a IA."
    ),
    "Pergunta a colegas": (
        "Pergunto a amigos e colegas"
    ),
}

q18_normalizada = dados[col_q18].fillna("").apply(normalizar)

# Matriz binária: 1 = participante selecionou a alternativa.
matriz_q18 = pd.DataFrame(index=dados.index)

for nome, alternativa in opcoes_q18.items():
    alternativa_norm = normalizar(alternativa)
    matriz_q18[nome] = q18_normalizada.apply(
        lambda resposta: int(alternativa_norm in resposta)
    )

# ============================================================
# GRÁFICO 3 — DISTRIBUIÇÃO GERAL DA Q18
# ============================================================

print("\n[3/4] Gerando distribuição geral da Q18...")

frequencias_q18 = matriz_q18.sum().sort_values(ascending=True)
percentuais_q18 = (frequencias_q18 / len(dados)) * 100

tabela_geral_q18 = pd.DataFrame({
    "Frequência": frequencias_q18,
    "Percentual": percentuais_q18
})

tabela_geral_q18.to_csv(
    PASTA_SAIDA / "q18_distribuicao_geral.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))

barras = plt.barh(
    frequencias_q18.index,
    frequencias_q18.values
)

plt.title(
    "Estratégias adotadas quando a IA não consegue resolver um problema"
)
plt.xlabel("Número de participantes")
plt.ylabel("Estratégias")

for barra, frequencia, percentual in zip(
    barras,
    frequencias_q18.values,
    percentuais_q18.values
):
    plt.text(
        barra.get_width() + 0.3,
        barra.get_y() + barra.get_height() / 2,
        f"{frequencia} ({percentual:.1f}%)",
        va="center"
    )

plt.xlim(0, max(frequencias_q18.values) + 8)

salvar_grafico("q18_distribuicao_geral.png")


# ============================================================
# GRÁFICO 4 — Q18 × Q3
# Estratégias diante das limitações da IA × conhecimento
# ============================================================

print("\n[4/4] Gerando Q18 × Q3...")

# Percentual de participantes de cada nível que selecionou
# cada estratégia. Como é múltipla seleção, a soma pode passar
# de 100% dentro de um mesmo grupo.

dados_q18_q3 = matriz_q18.join(dados[col_q3])

tamanhos_grupos = dados_q18_q3.groupby(col_q3).size()

tabela_q18_q3 = (
    dados_q18_q3.groupby(col_q3)[list(opcoes_q18.keys())].mean()
    * 100
)

tabela_q18_q3 = tabela_q18_q3.reindex(ordem_q3)

tabela_q18_q3.to_csv(
    PASTA_SAIDA / "q18_por_conhecimento_q3_percentuais.csv",
    encoding="utf-8-sig"
)

ax = tabela_q18_q3.plot(
    kind="bar",
    figsize=(12, 7),
    colormap="tab10",
    width=0.8
)

ax.set_title(
    "Estratégias diante das limitações da IA por nível de conhecimento"
)
ax.set_xlabel("Nível de conhecimento em programação")
ax.set_ylabel("Participantes que selecionaram cada estratégia (%)")
ax.set_ylim(0, 110)
ax.legend(
    title="Estratégias",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
ax.tick_params(axis="x", rotation=0)

salvar_grafico("q18_por_conhecimento_q3.png")


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 65)
print("ANÁLISE CONCLUÍDA")
print("=" * 65)

print("\nArquivos gerados:")
for arquivo in sorted(PASTA_SAIDA.glob("*.png")):
    print(f"- {arquivo.name}")

print("\nTABELA — DISTRIBUIÇÃO GERAL DA Q18")
print(tabela_geral_q18.to_string())

print("\nTABELA — Q18 POR NÍVEL DE CONHECIMENTO")
print(tabela_q18_q3.round(2).to_string())

print("\nObservação: Q18 é de múltipla seleção.")
print("Os percentuais são calculados sobre participantes de cada grupo.")
print("Por isso, a soma das categorias pode ultrapassar 100%.")
