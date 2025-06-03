import torch
import torch.nn as nn
import superp_init as superp
import ann
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

    csv_file="World-Model/pendulum_dataset.csv"
    emb="World-Model/dinov2_embeddings"

    ############################################
    # number of mini_batches
    ############################################
    NUM_BATCHES = superp.EPOCHS // superp.BATCH_SIZE

    # train and return the learned model
    time_start_train = time.time()
    res = train.itr_train(barr_nn, ctrl_nn,  NUM_BATCHES, csv_file, emb) # train the barrier 
    time_end_train = time.time()
    
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")

        
    return barr_nn, ctrl_nn


if __name__ =="__main__":
     [barr_nn,ctrl_nn]=barr_ctrl_nn()
     torch.save(barr_nn,r'saved_weights/barr_nn')
     torch.save(ctrl_nn,r'saved_weights/ctrl_nn')
