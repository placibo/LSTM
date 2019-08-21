def output_cell(activation_matrix,parameters):
    #get hidden to output parameters
    how = parameters['how']
    
    #get outputs 
    output_matrix = np.matmul(activation_matrix,how)
    output_matrix = softmax(output_matrix)
    
    return output_matrix