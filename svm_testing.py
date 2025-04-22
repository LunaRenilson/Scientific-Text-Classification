import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('rslp')
dados = pd.read_csv('dados_v2.csv')

'''
A escolha de tokenizar as palavras (e não as frases) se deu porque as palavras, separadamente, aparentam trazer muito valor categórico às áreas da ciência 
'''

def vetorizar(coluna):
    # Removendo espaços e símbolos
    coluna = coluna.fillna('').str.replace('; ', ';', regex=False)
    coluna = coluna.str.replace(' ;', ';', regex=False) 
    coluna = coluna.str.replace(';', ' ', regex=False)
    coluna = coluna.str.replace(',', ' ', regex=False)

    # Tokenizando
    coluna = coluna.apply(word_tokenize)

    # Stopwords e Stemming
    stemmer = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))
    coluna = coluna.apply(
        lambda tokens: [stemmer.stem(t.lower()) for t in tokens if t.lower() not in stop_words]
    )

    # Juntando para vetorização
    coluna = coluna.apply(lambda tokens: " ".join(tokens))

    # Vetorizando
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(coluna)

    return X_tfidf, vectorizer

# tokenizando colunas de entrada
dados['titulo'] = vetorizar(dados['titulo'])
dados['resumo'] = vetorizar(dados['resumo'])
dados['palavrasChave'] = vetorizar(dados['palavrasChave'])

# Tokenizando coluna de saida
# Cada campo recebe um valor. Ex: f = 0, q = 1, b = 2, etc... 
encoder = LabelEncoder()
y_codificado = encoder.fit_transform(dados['areasCiencia'])