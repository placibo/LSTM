#backpropagation
def backward_propagation(batch_labels,embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache,parameters):
    #calculate output errors 
    output_error_cache,activation_error_cache = calculate_output_cell_error(batch_labels,output_cache,parameters)
    
    #to store lstm error for each time step
    lstm_error_cache = dict()
    
    #to store embeding errors for each time step
    embedding_error_cache = dict()
    
    # next activation error 
    # next cell error  
    #for last cell will be zero
    eat = np.zeros(activation_error_cache['ea1'].shape)
    ect = np.zeros(activation_error_cache['ea1'].shape)
    
    #calculate all lstm cell errors (going from last time-step to the first time step)
    for i in range(len(lstm_cache),0,-1):
        #calculate the lstm errors for this time step 't'
        pae,pce,ee,le = calculate_single_lstm_cell_error(activation_error_cache['ea'+str(i)],eat,ect,parameters,lstm_cache['lstm'+str(i)],cell_cache['c'+str(i)],cell_cache['c'+str(i-1)])
        
        #store the lstm error in dict
        lstm_error_cache['elstm'+str(i)] = le
        
        #store the embedding error in dict
        embedding_error_cache['eemb'+str(i-1)] = ee
        
        #update the next activation error and next cell error for previous cell
        eat = pae
        ect = pce
    
    
    #calculate output cell derivatives
    derivatives = dict()
    derivatives['dhow'] = calculate_output_cell_derivatives(output_error_cache,activation_cache,parameters)
    
    #calculate lstm cell derivatives for each time step and store in lstm_derivatives dict
    lstm_derivatives = dict()
    for i in range(1,len(lstm_error_cache)+1):
        lstm_derivatives['dlstm'+str(i)] = calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm'+str(i)],embedding_cache['emb'+str(i-1)],activation_cache['a'+str(i-1)])
    
    #initialize the derivatives to zeros 
    derivatives['dfgw'] = np.zeros(parameters['fgw'].shape)
    derivatives['digw'] = np.zeros(parameters['igw'].shape)
    derivatives['dogw'] = np.zeros(parameters['ogw'].shape)
    derivatives['dggw'] = np.zeros(parameters['ggw'].shape)
    
    #sum up the derivatives for each time step
    for i in range(1,len(lstm_error_cache)+1):
        derivatives['dfgw'] += lstm_derivatives['dlstm'+str(i)]['dfgw']
        derivatives['digw'] += lstm_derivatives['dlstm'+str(i)]['digw']
        derivatives['dogw'] += lstm_derivatives['dlstm'+str(i)]['dogw']
        derivatives['dggw'] += lstm_derivatives['dlstm'+str(i)]['dggw']
    
    return derivatives,embedding_error_cache