import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

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
    
def plot_metrics(y_true, y_pred):
    import matplotlib.pyplot as plt

    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    accuracy = (pd.Series(y_true) == pd.Series(y_pred)).mean()

    metrics_df = pd.DataFrame({
        'Classe': labels,
        'Precisão': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    metrics_df = metrics_df.set_index('Classe')
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(metrics_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Precisão, Recall e F1-Score por Classe\nAcurácia geral: {:.2f}%'.format(accuracy * 100))
    plt.savefig('LLM/metrics/classification_metrics.png')
    plt.close()

plot_metrics(real, prediction)

plot_confusion_matrix(real, prediction)