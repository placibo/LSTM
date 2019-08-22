#train function test
def train(train_dataset,iters=1000,batch_size=20):
    #initalize the parameters
    parameters = initialize_parameters()
    
    #initialize the V and S parameters for Adam
    V = initialize_V(parameters)
    S = initialize_S(parameters)
    
    #generate the random embeddings
    embeddings = np.random.normal(0,0.01,(len(vocab),input_units))
    
    #to store the Loss, Perplexity and Accuracy for each batch
    J = []
    P = []
    A = []
    
    
    for step in range(iters):
        #get batch dataset
        index = step%len(train_dataset)
        batches = train_dataset[index]
        
        #forward propagation
        embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache = forward_propagation(batches,parameters,embeddings)
        
        #calculate the loss, perplexity and accuracy
        perplexity,loss,acc = cal_loss_accuracy(batches,output_cache)
        
        #backward propagation
        derivatives,embedding_error_cache = backward_propagation(batches,embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache,parameters)
        
        #update the parameters
        parameters,V,S = update_parameters(parameters,derivatives,V,S,step)
        
        #update the embeddings
        embeddings = update_embeddings(embeddings,embedding_error_cache,batches)
        
        
        J.append(loss)
        P.append(perplexity)
        A.append(acc)
        
        #print loss, accuracy and perplexity
        if(step%1000==0):
            print("For Single Batch :")
            print('Step       = {}'.format(step))
            print('Loss       = {}'.format(round(loss,2)))
            print('Perplexity = {}'.format(round(perplexity,2)))
            print('Accuracy   = {}'.format(round(acc*100,2)))
            print()
    
    return embeddings, parameters,J,P,A
