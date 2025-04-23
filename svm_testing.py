import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem import RSLPStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import swifter

# separar dados de treino e teste
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # treinar modelo
from sklearn.metrics import accuracy_score, classification_report # avaliar modelo
from sklearn.model_selection import cross_val_score


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('rslp')
dados = pd.read_csv('dados_v2.csv')

'''
A escolha de tokenizar as palavras (e não as frases) se deu porque as palavras, separadamente, aparentam trazer muito valor categórico às áreas da ciência 
'''

def preProcessing(x_coluna):
    # Removendo espaços e símbolos
    x_coluna = x_coluna.fillna('').str.replace('; ', ';', regex=False)
    x_coluna = x_coluna.str.replace(' ;', ';', regex=False) 
    x_coluna = x_coluna.str.replace(';', ' ', regex=False)
    x_coluna = x_coluna.str.replace(',', ' ', regex=False)

    # Tokenizando
    x_coluna = x_coluna.apply(word_tokenize)

    # Stopwords e Stemming
    # stemmer = RSLPStemmer()
    
    stemmer = SnowballStemmer("portuguese")
    stop_words = set(stopwords.words('portuguese'))
    
    # x_coluna = x_coluna.apply(
    #     lambda tokens: [stemmer.stem(t.lower()) for t in tokens if t.lower() not in stop_words]
    # )
    # Paralelizando a tokenização
    x_coluna = x_coluna.swifter.progress_bar(desc=f"Processando {x_coluna.name}").apply(
        lambda tokens: [stemmer.stem(t.lower()) for t in tokens if t.lower() not in stop_words]
    )
    
    # Juntando para vetorização
    x_coluna = x_coluna.apply(lambda tokens: " ".join(tokens))

    # Retornando coluna preprocessada
    return x_coluna

# tokenizando colunas de entrada
X_titulo = preProcessing(dados['titulo'])
X_resumo = preProcessing(dados['resumo'])
X_palavras = preProcessing(dados['palavrasChave'])

X_concatenado = X_titulo + ' '  + ' ' + X_resumo + ' ' + X_palavras 

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_concatenado)

# Tokenizando coluna de saida
# Cada campo recebe um valor. Ex: f = 0, q = 1, b = 2, etc... 
encoder = LabelEncoder()
y_codificado = encoder.fit_transform(dados['areasCiencia'])

# Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_codificado, test_size=0.3, random_state=42)


# Treinando modelo
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = svm_linear.predict(X_test)

# Avaliando o modelo
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

