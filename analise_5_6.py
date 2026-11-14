import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 1. Carregar os dados
CANDIDATOS = [
    BASE_DIR / 'dados' / 'data.csv',
    BASE_DIR / 'data.csv',
    BASE_DIR / 'dados' / 'respostas_tcc.csv',
    BASE_DIR / 'respostas_tcc.csv',
]
ARQUIVO = next((caminho for caminho in CANDIDATOS if caminho.exists()), None)

if ARQUIVO is None:
    raise FileNotFoundError(
        'Não encontrei a base de dados. Coloque o CSV em ./dados/data.csv ou ./data.csv.'
    )

if ARQUIVO.suffix.lower() in {'.xlsx', '.xls'}:
    df = pd.read_excel(ARQUIVO)
else:
    df = pd.read_csv(ARQUIVO, encoding='utf-8-sig')


def localizar_coluna_por_numero(df_base, numero_questao):
    numero_texto = str(numero_questao)
    padroes = [
        f'{numero_texto})',
        f'{numero_texto} -',
        f'Q{numero_texto}',
        f'q{numero_texto}',
    ]

    for nome_coluna in df_base.columns:
        texto = str(nome_coluna).strip()
        texto_lower = texto.lower()
        if any(texto_lower.startswith(p.lower()) for p in padroes):
            return nome_coluna
        if re.search(rf'(?i)^\s*(?:q|quest(?:ã|a)o)?\s*{re.escape(numero_texto)}\b', texto):
            return nome_coluna
    raise KeyError(f'Não foi encontrada a coluna da questão {numero_questao}.')


q1_col = localizar_coluna_por_numero(df, 1)
q3_col = localizar_coluna_por_numero(df, 3)
q5_col = localizar_coluna_por_numero(df, 5)
q26_col = localizar_coluna_por_numero(df, 26)

# Criar ID de participante (P1 a Pn)
df = df.copy()
df['ID'] = [f'P{i + 1}' for i in range(len(df))]


def normalizar_texto(valor):
    if pd.isna(valor):
        return ''
    texto = str(valor).strip()
    return ' '.join(texto.split())


def resposta_invalida(valor):
    texto = normalizar_texto(valor).lower()
    if not texto:
        return True
    if re.fullmatch(r'[\W_]+', texto):
        return True
    texto_sem_pontuacao = re.sub(r'[^a-z0-9à-ÿ]+', '', texto)
    invalid_exact = {
        '',
        'naotemconsideracoesadicionais',
        'nao',
        'nada',
        'semcomentarios',
        'semconsideracoes',
        'semobservacoes',
        'semcomentario',
        'nenhum',
        'nenhuma',
        'n/a',
        'na',
        '--',
        '.',
        '...',
    }
    return texto_sem_pontuacao in invalid_exact or texto in {'', '.', '--'}


df['q26_clean'] = df[q26_col].map(normalizar_texto)
mask_invalid = df['q26_clean'].map(resposta_invalida)

df_excluded = df[mask_invalid].copy()
df_valid = df[~mask_invalid].copy()

print('=' * 70)
print(f'ARQUIVO: {ARQUIVO}')
print(f'TOTAL DE RESPONDENTES: {len(df)}')
print(f'PREENCHIDOS NO FORMULÁRIO (não vazios): {df[q26_col].notna().sum()}')
print(f'DESCARTADOS NA TRIAGEM: {len(df_excluded)}')
for _, row in df_excluded.iterrows():
    print(f"  - {row['ID']}: {repr(row[q26_col])}")
print(f'RELATOS SUBSTANTIVOS ANALISADOS: {len(df_valid)}')
print('=' * 70)

# 2. Mapeamento das categorias temáticas (por número do participante 1..n)
# Como um relato pode abordar benefícios e riscos ao mesmo tempo, as categorias não são mutuamente exclusivas.
themes_map = {
    'T1 - Aceleração e tutoria personalizada': [
        1, 2, 4, 8, 9, 10, 15, 16, 17, 18, 22, 24, 25, 26, 27,
        30, 31, 32, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49
    ],
    'T2 - Dependência, perda de retenção e atrofia': [
        1, 3, 8, 10, 11, 13, 15, 16, 20, 21, 27, 31, 36, 38, 39,
        40, 41, 43, 47, 48, 50
    ],
    'T3 - Uso crítico e condicionalidade': [
        7, 8, 9, 15, 16, 20, 23, 29, 30, 32, 36, 37, 40, 43, 44,
        45, 46, 47, 48
    ],
    'T4 - Deslocamento cognitivo (micro -> macro)': [
        6, 12, 21, 37, 40, 51
    ]
}

# Verificar cobertura (garantir que nenhum relato válido ficou sem categoria)
valid_nums = set(df_valid.index + 1)
assigned_nums = set(num for nums in themes_map.values() for num in nums)
unassigned = sorted(valid_nums - assigned_nums)
print(f'\nRelatos válidos sem categoria: {unassigned if unassigned else "Nenhum (100% cobertos)"}\n')

# 3. Resumo quantitativo das categorias
n_valid = len(df_valid)
summary_rows = []
for theme, nums in themes_map.items():
    count = len(nums)
    pct = (count / n_valid) * 100 if n_valid else 0.0
    ids_str = ', '.join(f'P{n}' for n in sorted(nums))
    summary_rows.append({
        'Categoria': theme,
        'n': count,
        '% (sobre base válida)': round(pct, 2),
        'Participantes': ids_str,
    })

df_summary = pd.DataFrame(summary_rows)
print('--- RESUMO DAS CATEGORIAS TEMÁTICAS ---')
print(df_summary[['Categoria', 'n', '% (sobre base válida)']].to_string(index=False))

# 4. Inspeção detalhada por categoria (para revisão manual)
print('\n' + '=' * 70)
print('INSPEÇÃO DETALHADA DOS RELATOS POR CATEGORIA')
print('=' * 70)
for theme, nums in themes_map.items():
    print(f"\n>>> {theme} (n={len(nums)} | {len(nums) / n_valid * 100:.2f}% sobre base válida)")
    for n in sorted(nums):
        row = df.iloc[n - 1]
        texto = ' '.join(str(row[q26_col]).split())
        print(f"  [{row['ID']} | {row[q1_col]} | {row[q3_col]} | {row[q5_col]}]: {texto}")

# 5. Cruzamento opcional: Incidência dos temas por Nível de Conhecimento (Q3)
print('\n' + '=' * 70)
print('INCIDÊNCIA DOS TEMAS POR NÍVEL DE CONHECIMENTO (Q3)')
print('=' * 70)

for theme, nums in themes_map.items():
    df_valid[theme] = (df_valid.index + 1).isin(nums)

ct_q3 = df_valid.groupby(q3_col)[list(themes_map.keys())].agg(['sum', 'mean'])
for col in themes_map.keys():
    ct_q3[(col, 'mean')] = (ct_q3[(col, 'mean')] * 100).round(2)
print(ct_q3)