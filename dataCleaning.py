import pandas as pd
import numpy as np
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

disciplinas = ['F', 'B', 'Q', 'C', 'A']
dados = pd.read_csv('dados.csv')
dados = dados.loc[dados['areasCiencia'].isin(disciplinas)]

dados['areasCiencia'].hist(bins=5)
plt.show()

# print(areaCiencia)