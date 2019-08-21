#calculate output cell derivatives
def calculate_output_cell_derivatives(output_error_cache,activation_cache,parameters):
    #to store the sum of derivatives from each time step
    dhow = np.zeros(parameters['how'].shape)
    
    batch_size = activation_cache['a1'].shape[0]
    
    #loop through the time steps 
    for i in range(1,len(output_error_cache)+1):
        #get output error
        output_error = output_error_cache['eo' + str(i)]
        
        #get input activation
        activation = activation_cache['a'+str(i)]
        
        #cal derivative and summing up!
        dhow += np.matmul(activation.T,output_error)/batch_size
        
    return dhow