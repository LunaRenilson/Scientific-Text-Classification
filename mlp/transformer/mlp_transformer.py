from sentence_transformers import SentenceTransformer

# Modelo pré-treinado em português
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

# Vetorização direta (já faz todo o pré-processamento)
vectors = model.encode(df['texto_completo'].tolist(), show_progress_bar=True)

