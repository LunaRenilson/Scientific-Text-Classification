import pandas as pd
import numpy as np
import seaborn as sns
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
    (~dados['resumo'].str.contains('\?', na=False)) &
    (~dados['palavrasChave'].str.contains('\?', na=False))
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

# plotando histograma
# sns.histplot(dados['areasCiencia'], bins=5)
# plt.show()

# gerando arquivo com dados filtrados
dados.to_csv('dados_v2.csv')