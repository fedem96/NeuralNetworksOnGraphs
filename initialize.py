import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.set_up_planetoid import set_up_planetoid
from GAT.set_up_gat import set_up_gat

if __name__ == '__main__':

    model = "planetoid"             # can be "planetoid", "cheb" ,"gcn", "gat"
    dataset = "cora"                # can be "cora", "pubmed", "citeseer"    
    
    epochs = 10
    val_period = 5                  # each epoch validation
    log = 1                         # every two epochs print train loss and acc

    data_seed = 0
    net_seed = 0
    tf.random.set_seed(net_seed)
    np.random.seed(data_seed)

    if model=="planetoid":
        modality = "I"          # can be T (transductive) or I (inductive)    
        set_up_planetoid(dataset, modality, epochs, val_period, log)
    
    elif model=='cheb':
        ...
    
    elif model=='gcn':
        ...

    elif model=='gat':
        set_up_gat(dataset, epochs, val_period, log)