import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Astronomia
# Biologia
# Física
# Geociências
# Geografia
# Educação Ambiental (apenas se for junto ao ensino de Biologia ou Ciências)
# Química
# Ciências
# Matemática
# Computação

disciplinas = ['f', 'b', 'q', 'c']
dados = pd.read_csv('dados.csv')
dados['areasCiencia'] = dados['areasCiencia'].fillna('')
dados['areasCiencia'] = dados['areasCiencia'].str.lower() # campos para minusculo

dados = dados[
    (~dados['areasCiencia'].str.contains(' ', na=False)) &
    (~dados['areasCiencia'].str.contains('/', na=False)) &
    (~dados['areasCiencia'].str.contains(',', na=False)) &
    (~dados['areasCiencia'].str.contains('-', na=False)) &
    (~dados['areasCiencia'].str.contains('_', na=False)) &
    (~dados['areasCiencia'].str.contains(';', na=False))
    ]

dados.loc[dados['areasCiencia'].str.startswith('fís'), 'areasCiencia'] = 'f' # Renomeando campos da fisica para f

dados.loc[
    (dados['areasCiencia'].str.startswith('q', na=False)) &
    (dados['areasCiencia'].str.len() < 8), 
    'areasCiencia'
    ] = 'q'

dados.loc[
    (dados['areasCiencia'].str.startswith('b', na=False)) &
    (dados['areasCiencia'].str.len() < 9), 
    'areasCiencia'
    ] = 'b'

dados.loc[
    (dados['areasCiencia'].str.startswith('ger', na=False)) &
    (dados['areasCiencia'].str.len() < 6), 
    'areasCiencia'
    ] = 'c'


dados = dados.loc[dados['areasCiencia'].isin(disciplinas)]

sns.histplot(dados['areasCiencia'], bins=5)
plt.show()

dados.to_csv('dados_limpos.csv')