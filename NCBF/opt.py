import torch
import torch.nn as nn
import torch.optim as optim
import superp_init as superp

def set_optimizer(barr_nn):

    optimizer = optim.Adam([{'params':barr_nn.parameters()}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    return optimizer



    