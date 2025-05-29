from transformer.TextClassifierMLP import ClassifierMLP

def main():
    clf = ClassifierMLP(data_path='assets/dados_v2.csv')
    
    clf.load_processed_data()
    # clf.preprocess()
    clf.build_model()
    history = clf.train(epochs=50)
    clf.plot_history(history)
    # clf.save_model()

if __name__ == '__main__':
    main() 