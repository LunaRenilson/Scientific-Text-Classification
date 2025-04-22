import pandas as pd
import matplotlib.pyplot as plt

disciplinas = ['f', 'b', 'q', 'c']
dados = pd.read_csv('dados.csv', usecols=['titulo', 'resumo', 'palavrasChave', 'areasCiencia'])

dados = dados.fillna('')  # limpando campos com NaN
dados = dados.apply(lambda col: col.str.lower()) # campos para minusculo

dados = dados[
    (~dados['areasCiencia'].str.contains(' ', na=False)) &
    (~dados['areasCiencia'].str.contains('/', na=False)) &
    (~dados['areasCiencia'].str.contains(',', na=False)) &
    (~dados['areasCiencia'].str.contains('-', na=False)) &
    (~dados['areasCiencia'].str.contains('_', na=False)) &
    (~dados['areasCiencia'].str.contains(';', na=False)) &
    (~dados['areasCiencia'].str.contains(r'\+', na=False)) &
    (~dados['resumo'].str.contains(r'\?', na=False)) &
    (~dados['palavrasChave'].str.contains(r'\?', na=False))
]


# Selecionando campos da fisica
dados.loc[
    dados['areasCiencia'].str.startswith('fís'), 'areasCiencia'
    ] = 'f' # Renomeando campos da fisica para f

# Selecionando campos da quimica
dados.loc[
    (dados['areasCiencia'].str.startswith('q', na=False)) &
    (dados['areasCiencia'].str.len() < 8), 
    'areasCiencia'
    ] = 'q'

# Selecionando campos da biologia
dados.loc[
    (dados['areasCiencia'].str.startswith('b', na=False)) &
    (dados['areasCiencia'].str.len() < 9), 
    'areasCiencia'
    ] = 'b'

# Selecionando campos da ciencias gerais
dados.loc[
    (dados['areasCiencia'].str.startswith('ger', na=False)) &
    (dados['areasCiencia'].str.len() < 6), 
    'areasCiencia'
    ] = 'c'


# Selecionando disciplinas filtradas acima
dados = dados.loc[dados['areasCiencia'].isin(disciplinas)]

dados.to_csv('dados_v2.csv')


# forgraph = dados = dados[
#     (dados['areasCiencia'].str.startswith('fís', na=False)) |
#     (dados['areasCiencia'].str.startswith('fis', na=False)) |
#     (dados['areasCiencia'].str.startswith('quí', na=False)) |
#     (dados['areasCiencia'].str.startswith('qui', na=False)) |
#     (dados['areasCiencia'].str.startswith('bio', na=False)) |
#     (dados['areasCiencia'].str.startswith('ger', na=False)) |
#     (dados['areasCiencia'] == 'f') |
#     (dados['areasCiencia'] == 'b') |
#     (dados['areasCiencia'] == 'q') | 
#     (dados['areasCiencia'] == 'c')
# ]


# freq = forgraph['areasCiencia'].value_counts()

# # Plotando como gráfico de barras
# freq.plot(kind='bar', edgecolor='black')

# plt.xlabel('Categoria')
# plt.ylabel('Frequência')
# plt.title('Frequência de cada categoria')
# plt.xticks(rotation=45)
# plt.savefig('assets/hist_after-filtering.png')
# plt.tight_layout()
# plt.show()