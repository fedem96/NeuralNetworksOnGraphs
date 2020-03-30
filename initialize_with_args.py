import os 
import sys
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.set_up_planetoid import set_up_planetoid
from GAT.set_up_gat import set_up_gat

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", default=5/6)
    parser.add_argument("--r2", default=5/6)
    parser.add_argument("--q",  default=10)
    parser.add_argument("--d",  default=3)
    parser.add_argument("--n1", default=200)
    parser.add_argument("--n2", default=200)
    parser.add_argument("--t1", default=20)
    parser.add_argument("--t2", default=20)  
 
    args = parser.parse_args()

    model = "planetoid"             # can be "planetoid", "cheb" ,"gcn", "gat"
    dataset = "cora"                # can be "cora", "pubmed", "citeseer"    
    
    epochs = 10
    val_period = 5                  # each epoch validation
    log = 1                         # every two epochs print train loss and acc
    pre_train_iters = 100           # graph context pretrain iterations


    seed = 1234

    if model=="planetoid":

        modality = "T"          # can be T (transductive) or I (inductive)    
        embedding_size = 50   
        args = {'r1': args.r1, 'r2': args.r2, 'q':args.q , 'd':args.d, 'n1':args.n1, 'n2':args.n2, 't1':args.t1, 't2':args.t2}
        set_up_planetoid(embedding_size, dataset, seed, modality, epochs, val_period, log, pre_train_iters, args)
    
    elif model=='cheb':
        ...
    
    elif model=='gcn':
        ...

    elif model=='gat':
        
        set_up_gat()

if __name__ == '__main__':

    main()