#calculate output cell errors
def calculate_output_cell_error(batch_labels,output_cache,parameters):
    #to store the output errors for each time step
    output_error_cache = dict()
    activation_error_cache = dict()
    how = parameters['how']
    
    #loop through each time step
    for i in range(1,len(output_cache)+1):
        #get true and predicted labels
        labels = batch_labels[i]
        pred = output_cache['o'+str(i)]
        
        #calculate the output_error for time step 't'
        error_output = pred - labels
        
        #calculate the activation error for time step 't'
        error_activation = np.matmul(error_output,how.T)
        
        #store the output and activation error in dict
        output_error_cache['eo'+str(i)] = error_output
        activation_error_cache['ea'+str(i)] = error_activation
        
    return output_error_cache,activation_error_cache