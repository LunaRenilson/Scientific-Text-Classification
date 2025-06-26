import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('assets/llama_respostas.csv')

real = df['areasCiencia']
prediction = df['llama_response']

accuracy = (real == prediction).mean() * 100

print(f"Acurácia: {accuracy:.2f}%")

def plot_confusion_matrix(y_true, y_pred):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true) | set(y_pred)),
                yticklabels=sorted(set(y_true) | set(y_pred)))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig('LLM/metrics/confusion_matrix.png')

plot_confusion_matrix(real, prediction)