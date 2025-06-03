import torch
import torch.nn as nn
import superp_init as superp
import ann
import data
import train
import time
import policy

torch.cuda.empty_cache()

def barr_ctrl_nn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generating training model
    barr_nn = ann.gen_nn(superp.N_H_B, superp.D_H_B, superp.DIM_S, 1, superp.BARR_ACT, superp.BARR_IN_BOUND, superp.BARR_OUT_BOUND) # generate the nn model for the barrier
    barr_nn = barr_nn.to(device)

    # Load reference policy model
    ctrl_nn = policy.PolicyNet(embed_dim=768, num_patches=256, action_dim=1, hidden_dim=256)
    ctrl_nn.load_state_dict(torch.load('runs/imitation_policy/model_final.pth', map_location=device))
    ctrl_nn = ctrl_nn.to(device)
    ctrl_nn.eval()

    # loading pre-trained model
    # if superp.FINE_TUNE == 1:
    #     barr_nn=torch.load('saved_weights/barr_nn') #stored weights/barr_nn_best_100_0
    #     ctrl_nn=torch.load('saved_weights/ctrl_nn') #stored weights/ctrl_nn_best_100_0
    
    
    # generate training data
    time_start_data = time.time()

    batches_init, batches_unsafe, batches_domain = data.gen_batch_data()

    time_end_data = time.time()
    
    ############################################
    # number of mini_batches
    ############################################
    BATCHES_I = len(batches_init)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_I, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_I, BATCHES_U, BATCHES_D, BATCHES]
    
    # train and return the learned model
    time_start_train = time.time()
    res = train.itr_train(barr_nn, ctrl_nn, batches_init, batches_unsafe, batches_domain, NUM_BATCHES) 
    time_end_train = time.time()
    
    print("\nData generation totally costs:", time_end_data - time_start_data)
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")

        
    return barr_nn, ctrl_nn


if __name__ =="__main__":
     [barr_nn,ctrl_nn]=barr_ctrl_nn()
     torch.save(barr_nn,r'saved_weights/barr_nn')
     torch.save(ctrl_nn,r'saved_weights/ctrl_nn')
