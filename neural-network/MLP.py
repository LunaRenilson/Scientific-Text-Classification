# Libs de treino
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Libs de pre-processamento
from gensim.models.fasttext import load_facebook_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.metrics import Precision, Recall


# Carrega os dados
df = pd.read_csv('assets/svm/dados_processados.csv')

# # Unindo os campos
# df['texto_completo'] = (
#     df['titulo_processado'].fillna('') + ' ' +
#     df['resumo_processado'].fillna('') + ' ' +
#     df['palavrasChave'].fillna('')
# )
from sklearn.feature_extraction.text import TfidfVectorizer

# Escolhendo os 5k termos mais frequentes
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_concatenado']).toarray()

# Converte rótulos para números inteiros
le = LabelEncoder()
y_int = le.fit_transform(df['classes_originais'])  # f → 0, q → 1, etc.

# Converte para one-hot
y = to_categorical(y_int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = build_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()