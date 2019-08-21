#update the parameters using adam optimizer
#adam optimization
def update_parameters(parameters,derivatives,V,S,t):
    #get derivatives
    dfgw = derivatives['dfgw']
    digw = derivatives['digw']
    dogw = derivatives['dogw']
    dggw = derivatives['dggw']
    dhow = derivatives['dhow']
    
    #get parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    ggw = parameters['ggw']
    how = parameters['how']
    
    #get V parameters
    vfgw = V['vfgw']
    vigw = V['vigw']
    vogw = V['vogw']
    vggw = V['vggw']
    vhow = V['vhow']
    
    #get S parameters
    sfgw = S['sfgw']
    sigw = S['sigw']
    sogw = S['sogw']
    sggw = S['sggw']
    show = S['show']
    
    #calculate the V parameters from V and current derivatives
    vfgw = (beta1*vfgw + (1-beta1)*dfgw)
    vigw = (beta1*vigw + (1-beta1)*digw)
    vogw = (beta1*vogw + (1-beta1)*dogw)
    vggw = (beta1*vggw + (1-beta1)*dggw)
    vhow = (beta1*vhow + (1-beta1)*dhow)
    
    #calculate the S parameters from S and current derivatives
    sfgw = (beta2*sfgw + (1-beta2)*(dfgw**2))
    sigw = (beta2*sigw + (1-beta2)*(digw**2))
    sogw = (beta2*sogw + (1-beta2)*(dogw**2))
    sggw = (beta2*sggw + (1-beta2)*(dggw**2))
    show = (beta2*show + (1-beta2)*(dhow**2))
    
    #update the parameters
    fgw = fgw - learning_rate*((vfgw)/(np.sqrt(sfgw) + 1e-6))
    igw = igw - learning_rate*((vigw)/(np.sqrt(sigw) + 1e-6))
    ogw = ogw - learning_rate*((vogw)/(np.sqrt(sogw) + 1e-6))
    ggw = ggw - learning_rate*((vggw)/(np.sqrt(sggw) + 1e-6))
    how = how - learning_rate*((vhow)/(np.sqrt(show) + 1e-6))
    
    #store the new weights
    parameters['fgw'] = fgw
    parameters['igw'] = igw
    parameters['ogw'] = ogw
    parameters['ggw'] = ggw
    parameters['how'] = how
    
    #store the new V parameters
    V['vfgw'] = vfgw 
    V['vigw'] = vigw 
    V['vogw'] = vogw 
    V['vggw'] = vggw
    V['vhow'] = vhow
    
    #store the s parameters
    S['sfgw'] = sfgw 
    S['sigw'] = sigw 
    S['sogw'] = sogw 
    S['sggw'] = sggw
    S['show'] = show
    
    return parameters,V,S