#forward propagation
def forward_propagation(batches,parameters,embeddings):
    #get batch size
    batch_size = batches[0].shape[0]
    
    #to store the activations of all the unrollings.
    lstm_cache = dict()                 #lstm cache
    activation_cache = dict()           #activation cache 
    cell_cache = dict()                 #cell cache
    output_cache = dict()               #output cache
    embedding_cache = dict()            #embedding cache 
    
    #initial activation_matrix(a0) and cell_matrix(c0)
    a0 = np.zeros([batch_size,hidden_units],dtype=np.float32)
    c0 = np.zeros([batch_size,hidden_units],dtype=np.float32)
    
    #store the initial activations in cache
    activation_cache['a0'] = a0
    cell_cache['c0'] = c0
    
    #unroll the names
    for i in range(len(batches)-1):
        #get first first character batch
        batch_dataset = batches[i]
        
        #get embeddings 
        batch_dataset = get_embeddings(batch_dataset,embeddings)
        embedding_cache['emb'+str(i)] = batch_dataset
        
        #lstm cell
        lstm_activations,ct,at = lstm_cell(batch_dataset,a0,c0,parameters)
        
        #output cell
        ot = output_cell(at,parameters)
        
        #store the time 't' activations in caches
        lstm_cache['lstm' + str(i+1)]  = lstm_activations
        activation_cache['a'+str(i+1)] = at
        cell_cache['c' + str(i+1)] = ct
        output_cache['o'+str(i+1)] = ot
        
        #update a0 and c0 to new 'at' and 'ct' for next lstm cell
        a0 = at
        c0 = ct
        
    return embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache