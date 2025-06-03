import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import superp_init as superp # parameters
import loss # computing loss
import opt
import lrate
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_with_loader
torch.set_printoptions(precision=7)

#################################################
# iterative training: the most important function
# it relies on three assistant functions:
#################################################

# used to output learned model parameters
def print_nn(model):
    for p in model.parameters():
        print(p.data)

def print_nn_matlab(model):
    layer = 0
    for p in model.parameters():
        layer = layer + 1
        arr = p.detach().numpy()
        if arr.ndim == 2:
            print( "w" + str((layer + 1) // 2) + " = [", end="")
            print('; '.join([', '.join(str(curr_int) for curr_int in curr_arr) for curr_arr in arr]), end="];\n")
        elif arr.ndim == 1:
            print( "b" + str(layer // 2) + " = [", end="")
            if layer == 2:
                print(', '.join(str(i) for i in arr), end="]';\n")
            else:
                print(', '.join(str(i) for i in arr), end="];\n")
        else:
            print("Transform error!")
    
def initialize_nn(barr_nn, ctrl_nn, num_batches):    
    print("Initialize nn parameters!")

    ## random initialize or load saved
    if superp.FINE_TUNE == 0:
        for p in barr_nn.parameters():
            nn.init.normal_(p,0,0.001) #standard Gaussian distribution


    else:
        for p in barr_nn.parameters():
            p.requires_grad=True

    optimizer = opt.set_optimizer(barr_nn)
    scheduler = lrate.set_scheduler(optimizer, num_batches)

    return optimizer,scheduler

def itr_train(barr_nn, ctrl_nn, NUM_BATCHES, csv, emb): 
    print("AAAAAAAAAAAAAAAAAAAAA", csv)
    dataset=data_with_loader.dataset(csv_file= csv, embedding_dir=emb)
    dataloader=DataLoader(dataset=dataset, batch_size=superp.BATCH_SIZE, shuffle=True)

    ############################## the main training loop ##################################################################
    best_loss = float('inf')
    best_barr_path = "best_barr_nn.pth"

    
    # initialize nn models and optimizers and schedulers
    optimizer_barr, scheduler_barr = initialize_nn(barr_nn, NUM_BATCHES)

    for epoch in tqdm(range(superp.EPOCHS), desc='Epochs'): # train for a number of epochs
        epoch_loss = 0 # scalar (move outside mini-batch loop)
        
        for i, (inputs,labels) in enumerate(dataloader):# initialize epoch
            lmi_loss = 0 #scalar
            eta_loss = 0
            epoch_gradient_flag = True # gradient is within range
            superp.CURR_MAX_GRAD = 0

                ############################## mini-batch training ################################################
            optimizer_barr.zero_grad() # clear gradient of parameters
            
                

            curr_batch_loss = loss.calc_loss(barr_nn, ctrl_nn, inputs,labels) 
                # batch_loss is a tensor, batch_gradient is a scalar
            curr_batch_loss.backward() # compute gradient using backward()
                # update weight and bias
            optimizer_barr.step() # gradient descent once
                
            optimizer_barr.zero_grad()
                                            
            # learning rate scheduling for each mini batch
            scheduler_barr.step() # re-schedule learning rate once


            # update epoch loss
            epoch_loss += curr_batch_loss.item()
            
    

            if superp.VERBOSE == 1:
                print("epoch: %-3s" % epoch, "batch_loss: %-25s" % curr_batch_loss.item(), \
                        "epoch_loss: %-25s" % epoch_loss, "lmi loss: % 25s" %lmi_loss)
        # Save model if this epoch's loss is the best so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(barr_nn.state_dict(), best_barr_path)
            if superp.VERBOSE == 1:
                print(f"Saved new best models at epoch {epoch} with loss {best_loss}")

            

    return False


