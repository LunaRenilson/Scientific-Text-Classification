import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# separar dados de treino e teste
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # treinar modelo
from sklearn.metrics import accuracy_score, classification_report # avaliar modelo
from sklearn.model_selection import cross_val_score



dados = pd.read_csv('dados_processados.csv')

# tokenizando colunas de entrada
# X_resumo = dados['resumo_processado']
# X_titulo = dados['titulo_processado']
# X_palavras = dados['palavrasChave_processado']

X_concatenado = (
    dados['titulo_processado'].fillna('') + ' ' +
    dados['resumo_processado'].fillna('') + ' ' +
    dados['palavrasChave_processado'].fillna('')
)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_concatenado)

# Tokenizando coluna de saida
# Cada campo recebe um valor. Ex: f = 0, q = 1, b = 2, etc... 
encoder = LabelEncoder()
y_codificado = encoder.fit_transform(dados['classes_originais'])

# Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_codificado, test_size=0.3, random_state=42,)

for i in range(30):
    # Treinando modelo
    svm_linear = SVC(
        kernel='linear', 
        random_state=42, 
        C=i+1,
        class_weight='balanced'
    )

    svm_linear.fit(X_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = svm_linear.predict(X_test)

    # Avaliando o modelo
    # print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"c = {i+1}: {round(accuracy_score(y_test, y_pred), 2)}")