#single lstm cell
def lstm_cell(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
    #get parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    ggw = parameters['ggw']
    
    #concat batch data and prev_activation matrix
    concat_dataset = np.concatenate((batch_dataset,prev_activation_matrix),axis=1)
    
    #forget gate activations
    fa = np.matmul(concat_dataset,fgw)
    fa = sigmoid(fa)
    
    #input gate activations
    ia = np.matmul(concat_dataset,igw)
    ia = sigmoid(ia)
    
    #output gate activations
    oa = np.matmul(concat_dataset,ogw)
    oa = sigmoid(oa)
    
    #gate gate activations
    ga = np.matmul(concat_dataset,ggw)
    ga = tanh_activation(ga)
    
    #new cell memory matrix
    cell_memory_matrix = np.multiply(fa,prev_cell_matrix) + np.multiply(ia,ga)
    
    #current activation matrix
    activation_matrix = np.multiply(oa, tanh_activation(cell_memory_matrix))
    
    #lets store the activations to be used in back prop
    lstm_activations = dict()
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga
    
    return lstm_activations,cell_memory_matrix,activation_matrix