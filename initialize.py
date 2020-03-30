import os 
import sys

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.set_up_planetoid import set_up_planetoid
from GAT.set_up_gat import set_up_gat


if __name__ == '__main__':

    model = "planetoid"             # can be "planetoid", "cheb" ,"gcn", "gat"
    dataset = "cora"                # can be "cora", "pubmed", "citeseer"    
    
    epochs = 10
    val_period = 5                  # each epoch validation
    log = 1                         # every two epochs print train loss and acc
    pre_train_iters = 100           # graph context pretrain iterations

    if model=="planetoid":

        modality = "I"          # can be T (transductive) or I (inductive)    
        embedding_size = 50   
        seed = 1234
        args = {'r1': 5/6, 'r2': 5/6, 'q':10 , 'd':3, 'n1':200, 'n2':200, 't1':20, 't2':20}
        set_up_planetoid(embedding_size, dataset, seed, modality, epochs, val_period, log, pre_train_iters, args)
    
    elif model=='cheb':
        ...
    
    elif model=='gcn':
        ...

    elif model=='gat':
        set_up_gat()