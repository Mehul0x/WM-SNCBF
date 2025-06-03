import torch
import torch.nn as nn
import superp_init as superp
import prob
import torch.nn.functional as F
############################################
# constraints for barrier certificate B:
# (1) init ==> B <= 0
# (2) unsafe ==> B >= lambda 
# where lambda>0
# (3) domain ==> B(f(x))-B <= 0
############################################


############################################
# given the training data, compute the loss
############################################
    
    
def calc_loss(barr_nn, ctrl_nn, inputs, labels):
    # compute loss of init        
    loss_init=torch.tensor(0.0)
    loss_unsafe=torch.tensor(0.0)
    loss_lie=torch.tensor(0.0)
    total_loss = torch.tensor(0.0)

    for i, input in enumerate(inputs):
        if labels[i] == 'safe':
            output_init = barr_nn(input)
            loss_init = torch.relu(output_init - superp.gamma + superp.TOL_INIT ) #tolerance
        elif labels[i] == 'unsafe':
            output_unsafe = barr_nn(input)
            loss_unsafe = torch.relu((- output_unsafe) + superp.lamda + superp.TOL_UNSAFE) #tolerance

        # compute loss of domain
        output_domain=barr_nn(input)
    
        vector_domain = prob.vector_field(input, ctrl_nn) # compute vector field at domain
        output_vector=barr_nn(vector_domain)
        loss_lie=torch.relu(output_vector-output_domain + superp.TOL_LIE )
            
        total_loss = superp.DECAY_INIT * torch.sum(loss_init) + superp.DECAY_UNSAFE * torch.sum(loss_unsafe) \
                    + superp.DECAY_LIE * torch.sum(loss_lie) #+ loss_eta 
                    
    # return total_loss is a tensor, max_gradient is a scalar
    return total_loss


    

    
    
