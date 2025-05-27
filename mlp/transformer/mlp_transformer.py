
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

from tensorflow.keras.metrics import Precision, Recall
from sentence_transformers import SentenceTransformer

# Carrega os dados
df = pd.read_csv('assets/dados_v2.csv')

# Modelo pré-treinado em português
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

# Vetorização direta (já faz todo o pré-processamento)
df['texto_completo'] = df['titulo'] + ' ' + df['resumo'] + ' '+ df['palavrasChave']

X = model.encode(df['texto_completo'].tolist(), show_progress_bar=True)

# Converte rótulos para números inteiros
le = LabelEncoder()
y_int = le.fit_transform(df['classes_originais'])  # f → 0, q → 1, etc.

# Converte para one-hot
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

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = build_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop]
)

def plot_history(history):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Transforma em lista 1D para iteração

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
    plt.savefig('mlp/transformer/mlp_métricas-transformer.png')
    
plot_history(history)
model.save('mlp/transformer/MLP_Transformer.h5')

