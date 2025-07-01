# Libs de treino
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Libs de pre-processamento
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

from tensorflow.keras.metrics import Precision, Recall

# Carrega os dados
df = pd.read_csv('assets/dados_processados.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

# Escolhendo os 5k termos mais frequentes
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_concatenado']).toarray()

# Converte rótulos para números inteiros
le = LabelEncoder()
y_int = le.fit_transform(df['classes_originais'])  # f → 0, q → 1, etc.

# Converte para one-hotclear
y = to_categorical(y_int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()])
    
    return model

def plot_history(history, model, X_val, y_val, label_encoder):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history.history[metric], label=f'Train {metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('mlp/tf-idf/mlp_métricas-tfidf.png')
    plt.close()

    # Previsões no conjunto de validação
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    labels = label_encoder.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("mlp/tf-idf/mlp_matriz_confusao.png")
    plt.close()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = build_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop]
)

plot_history(history, model, X_val, y_val, le)