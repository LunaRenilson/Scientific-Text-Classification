import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall


class ClassifierMLP:
    def __init__(self, data_path, save_dir='mlp/transformer/preprocessed', transformer_model='neuralmind/bert-base-portuguese-cased'):
        self.data_path = data_path
        self.save_dir = save_dir
        self.transformer_model = transformer_model
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = None
        self.model = None
        os.makedirs(save_dir, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['texto_completo'] = (
            self.df['titulo'].fillna('') + ' ' +
            self.df['resumo'].fillna('') + ' ' +
            self.df['palavrasChave'].fillna('')
        )

    def preprocess(self, force_recompute=False):
        x_path = os.path.join(self.save_dir, 'X.npy')
        y_path = os.path.join(self.save_dir, 'y.npy')
        le_path = os.path.join(self.save_dir, 'label_encoder.joblib')

        if not force_recompute and all(map(os.path.exists, [x_path, y_path, le_path])):
            self.X = np.load(x_path)
            self.y = np.load(y_path)
            self.label_encoder = joblib.load(le_path)
            print("[INFO] Dados pré-processados carregados do disco.")
            return

        print("[INFO] Gerando embeddings com SentenceTransformer...")
        model = SentenceTransformer(self.transformer_model)
        texts = self.df['texto_completo'].astype(str).tolist()
        self.X = model.encode(texts, show_progress_bar=True)

        self.label_encoder = LabelEncoder()
        y_int = self.label_encoder.fit_transform(self.df['areasCiencia'])
        self.y = to_categorical(y_int)

        # Salva
        np.save(x_path, self.X)
        np.save(y_path, self.y)
        joblib.dump(self.label_encoder, le_path)
        print("[INFO] Pré-processamento salvo.")

    def build_model(self):
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(self.X.shape[1],), kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

    def train(self, epochs=25, batch_size=32, validation_split=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=validation_split, stratify=self.y
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )
        return history

    def save_history(self, history, output_path='mlp/transformer/mlp_métricas-transformer.png'):
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
        plt.savefig(output_path)
        print(f"[INFO] Gráfico salvo em {output_path}")
        
    def plot_history(self, history, output_path='mlp/transformer/mlp_métricas-transformer.png'):
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
        plt.show()
        

    def save_model(self, path='mlp/transformer/MLP_Transformer.h5'):
        self.model.save(path)
        print(f"[INFO] Modelo salvo em {path}")
        
    def load_processed_data(self):
        self.X = np.load(os.path.join(self.save_dir, 'X.npy'))
        self.y = np.load(os.path.join(self.save_dir, 'y.npy'))
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            stratify=self.y
        )
