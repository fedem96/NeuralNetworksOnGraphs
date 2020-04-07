import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from add_parent_path import add_parent_path

from models import GAT

with add_parent_path():
    from metrics import masked_accuracy, masked_loss
    from utils import *

def main(dataset_name, 
        nheads, hidden_units, feat_drop_rate, 
        coefs_drop_rate, l2_weight,
        data_seed, net_seed, checkpoint_path):

    # reproducibility
    np.random.seed(data_seed)
    tf.random.set_seed(net_seed)

    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))

    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    
    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)
    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    print("calculating adjacency matrix")
    graph = adjacency_matrix(neighbors)

    print("defining model")
    model = GAT(graph, num_classes, hidden_units, nheads, feat_drop_rate, coefs_drop_rate)

    model.compile(loss=lambda y_true, y_pred: masked_loss(y_true, y_pred) + l2_weight * tf.nn.l2_loss(y_pred-y_true), 
                    metrics=[masked_accuracy])


    print("load model from checkpoint")
    model.load_weights(checkpoint_path+'GAT_ckpts/cp.ckpt')

    print("test the model on test set")
    model.evaluate(features, y_test, batch_size=len(features))    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GAT')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="citeseer", choices=["citeseer", "cora", "pubmed"])
    
    # network hyperparameters
    parser.add_argument('-nh', '--nheads', help='heads number per layer (the len of the list represent the model layers number)', default='8,1')
    parser.add_argument("-hu", "--hidden-units", help="number of Graph Convolutional filters in the first layer", default=8, type=int)
    parser.add_argument("-fd", "--feat-drop-rate", help="dropout rate for model dropout layers (fraction of the input units to drop)", default=0.4, type=float)
    parser.add_argument("-cd", "--coefs-drop-rate", help="dropout rate for attention coefficients (fraction of the input units to drop)", default=0.4, type=float)

    # optimization parameters
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)
 
    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for model checkpoints")

    args = parser.parse_args()
    nheads = [int(item) for item in args.nheads.split(',')]
    
    main(args.dataset, 
        nheads, args.hidden_units, args.feat_drop_rate, 
        args.coefs_drop_rate, args.l2_weight, 
        args.data_seed, args.net_seed, args.checkpoint_path)
