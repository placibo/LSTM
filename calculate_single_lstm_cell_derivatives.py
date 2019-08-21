#calculate derivatives for single lstm cell
def calculate_single_lstm_cell_derivatives(lstm_error,embedding_matrix,activation_matrix):
    #get error for single time step
    ef = lstm_error['ef']
    ei = lstm_error['ei']
    eo = lstm_error['eo']
    eg = lstm_error['eg']
    
    #get input activations for this time step
    concat_matrix = np.concatenate((embedding_matrix,activation_matrix),axis=1)
    
    batch_size = embedding_matrix.shape[0]
    
    #cal derivatives for this time step
    dfgw = np.matmul(concat_matrix.T,ef)/batch_size
    digw = np.matmul(concat_matrix.T,ei)/batch_size
    dogw = np.matmul(concat_matrix.T,eo)/batch_size
    dggw = np.matmul(concat_matrix.T,eg)/batch_size
    
    #store the derivatives for this time step in dict
    derivatives = dict()
    derivatives['dfgw'] = dfgw
    derivatives['digw'] = digw
    derivatives['dogw'] = dogw
    derivatives['dggw'] = dggw
    
    return derivatives