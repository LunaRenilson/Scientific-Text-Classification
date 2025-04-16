import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download ('punkt_tab')


dados = pd.read_csv('dados_v2.csv')


'''
A escolha de tokenizar as palavras (e não as frases) se deu porque as palavras, separadamente, aparentam trazer muito valor categórico às áreas da ciência 
'''
# Removendo espaços entre separação das palavras-chave
dados['palavrasChave'] = dados['palavrasChave'].fillna('').str.replace('; ', ';', regex=False)
dados['palavrasChave'] = dados['palavrasChave'].fillna('').str.replace(' ;', ';', regex=False) 

# trocando separação das palavras-chave ";" -> " "
dados['palavrasChave'] = dados['palavrasChave'].str.replace(';', ' ', regex=False)

# Removendo virgulas nas palavras-chave
dados['palavrasChave'] = dados['palavrasChave'].str.replace(',', ' ', regex=False)

# Tokenizando as palavras
dados['palavrasChave'] = dados['palavrasChave'].apply(word_tokenize)


# Removendo stopwords: "do", "de", "no", "na", etc. 
# Lista de stopwords em português
# Aplicando Stemming nos tokens
stemmer = PorterStemmer()
stop_words = set(stopwords.words('portuguese'))
# Removendo stopwords do texto
dados['palavrasChave'] = dados['palavrasChave'].apply(
    lambda tokens: [stemmer.stem(t) for t in tokens if t.lower() not in stop_words]
)

print("Tokens filtrados:", dados['palavrasChave'])


# stems = dados['palavrasChave'].apply(
#     lambda tokens: [stemmer.stem(t) for t in dados['palavrasChave']]
# )

print("Stemming:", dados['palavrasChave'])


# Tokenização
# palavras = word_tokenize(texto)
# sentencas = sent_tokenize(texto)
# print("Palavras: ", palavras)