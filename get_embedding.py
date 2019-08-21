def get_embeddings(batch_dataset,embeddings):
    embedding_dataset = np.matmul(batch_dataset,embeddings)
    return embedding_dataset