import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Graficos das metricas
import matplotlib.pyplot as plt
import seaborn as sns

# separar dados de treino e teste
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # treinar modelo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # avaliar modelo
from sklearn.model_selection import cross_val_score
import joblib  # Importa joblib para salvar modelos

# Matriz de Confusão
def plot_confusion_matrix(y_true, y_pred, classes, title='Matriz de Confusão'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig('assets/svm/Matriz_Confusao.png')

# Plot Classification Report
def plot_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(2)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap='Blues', 
                cbar=False, linewidths=0.5)
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig('assets/svm/Classification_Report.png')


dados = pd.read_csv('assets/svm/dados_processados.csv')


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


melhor_modelo = None
melhor_score = 0
melhor_c = None

svm_linear = SVC(
    kernel='linear',
    random_state=42,
    C=1,
    class_weight='balanced'
)

svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_test)
score = accuracy_score(y_test, y_pred)

plot_confusion_matrix(y_test, y_pred, encoder.classes_)
plot_classification_report(y_test, y_pred, encoder.classes_)

# Salvando o modelo
joblib.dump(melhor_modelo, f"svm/model/svm_linear_C{melhor_c}.joblib")
# Salvando o vectorizer e o encoder
joblib.dump(vectorizer, "svm/model/vectorizer.joblib")
joblib.dump(encoder, "svm/model/label_encoder.joblib")
