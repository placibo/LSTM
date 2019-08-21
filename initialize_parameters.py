#initialize parameters
def initialize_parameters():
    #initialize the parameters with 0 mean and 0.01 standard deviation
    mean = 0
    std = 0.01
    
    #lstm cell weights
    forget_gate_weights = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
    input_gate_weights  = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
    output_gate_weights = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
    gate_gate_weights   = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
    
    #hidden to output weights (output cell)
    hidden_output_weights = np.random.normal(mean,std,(hidden_units,output_units))
    
    parameters = dict()
    parameters['fgw'] = forget_gate_weights
    parameters['igw'] = input_gate_weights
    parameters['ogw'] = output_gate_weights
    parameters['ggw'] = gate_gate_weights
    parameters['how'] = hidden_output_weights
    
    return parameters

def initialize_V(parameters):
    Vfgw = np.zeros(parameters['fgw'].shape)
    Vigw = np.zeros(parameters['igw'].shape)
    Vogw = np.zeros(parameters['ogw'].shape)
    Vggw = np.zeros(parameters['ggw'].shape)
    Vhow = np.zeros(parameters['how'].shape)
    
    V = dict()
    V['vfgw'] = Vfgw
    V['vigw'] = Vigw
    V['vogw'] = Vogw
    V['vggw'] = Vggw
    V['vhow'] = Vhow
    return V

def initialize_S(parameters):
    Sfgw = np.zeros(parameters['fgw'].shape)
    Sigw = np.zeros(parameters['igw'].shape)
    Sogw = np.zeros(parameters['ogw'].shape)
    Sggw = np.zeros(parameters['ggw'].shape)
    Show = np.zeros(parameters['how'].shape)
    
    S = dict()
    S['sfgw'] = Sfgw
    S['sigw'] = Sigw
    S['sogw'] = Sogw
    S['sggw'] = Sggw
    S['show'] = Show
    return S