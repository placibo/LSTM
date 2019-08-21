#calculate loss, perplexity and accuracy
def cal_loss_accuracy(batch_labels,output_cache):
    loss = 0  #to sum loss for each time step
    acc  = 0  #to sum acc for each time step 
    prob = 1  #probability product of each time step predicted char
    
    #batch size
    batch_size = batch_labels[0].shape[0]
    
    #loop through each time step
    for i in range(1,len(output_cache)+1):
        #get true labels and predictions
        labels = batch_labels[i]
        pred = output_cache['o'+str(i)]
        
        prob = np.multiply(prob,np.sum(np.multiply(labels,pred),axis=1).reshape(-1,1))
        loss += np.sum((np.multiply(labels,np.log(pred)) + np.multiply(1-labels,np.log(1-pred))),axis=1).reshape(-1,1)
        acc  += np.array(np.argmax(labels,1)==np.argmax(pred,1),dtype=np.float32).reshape(-1,1)
    
    #calculate perplexity loss and accuracy
    perplexity = np.sum((1/prob)**(1/len(output_cache)))/batch_size
    loss = np.sum(loss)*(-1/batch_size)
    acc  = np.sum(acc)/(batch_size)
    acc = acc/len(output_cache)
    
    return perplexity,loss,acc