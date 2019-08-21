#update the Embeddings
def update_embeddings(embeddings,embedding_error_cache,batch_labels):
    #to store the embeddings derivatives
    embedding_derivatives = np.zeros(embeddings.shape)
    
    batch_size = batch_labels[0].shape[0]
    
    #sum the embedding derivatives for each time step
    for i in range(len(embedding_error_cache)):
        embedding_derivatives += np.matmul(batch_labels[i].T,embedding_error_cache['eemb'+str(i)])/batch_size
    
    #update the embeddings
    embeddings = embeddings - learning_rate*embedding_derivatives
    return embeddings