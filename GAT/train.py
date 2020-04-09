import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from add_parent_path import add_parent_path

from models import GAT

with add_parent_path():
    from metrics import masked_accuracy, masked_loss, EarlyStoppingAccLoss
    from utils import *

def main(dataset_name, 
        nheads, hidden_units, feat_drop_rate, coefs_drop_rate,
        epochs, learning_rate, l2_weight, patience, 
        data_seed, net_seed, checkpoint_path, verbose):

    # reproducibility
    np.random.seed(data_seed)
    tf.random.set_seed(net_seed)

    if verbose > 0: print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))

    if verbose > 0: print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    features = normalize_features(features)
    
    if verbose > 0: print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)
    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    if verbose > 0: print("calculating adjacency matrix")
    graph = adjacency_matrix(neighbors)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if verbose > 0: print("defining model")
    model = GAT(graph, num_classes, hidden_units, nheads, feat_drop_rate, coefs_drop_rate)

    model.compile(loss=lambda y_true, y_pred: masked_loss(y_true, y_pred) + l2_weight * tf.nn.l2_loss(y_true-y_pred), 
                    optimizer=optimizer, metrics=[masked_accuracy])


    if verbose > 0: print("begin training")    
    tb = TensorBoard(log_dir='logs') #TODO: change dir
    es = EarlyStoppingAccLoss(patience, checkpoint_path, 'GAT')

    model.fit(features, y_train, epochs=epochs, batch_size=len(features), shuffle=False, validation_data=(features, y_val), callbacks=[tb, es], verbose=verbose)

    print('\nbest val_acc {:.3f} val_loss {:.3f} \n' .format(es.stopped_epoch, es.best_a, es.best_l))
    best_a = es.best_a
    best_l = es.best_l


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
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10, type=int)  
    parser.add_argument("-lr", "--learning-rate", help="starting learning rate of Adam optimizer", default=5e-3, type=float)
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)
    parser.add_argument("-p", "--patience", help="patience for early stop", default=100, type=int)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for model checkpoints", default=None)
    
    # verbose
    parser.add_argument("-v", "--verbose", help="useful print", default=1, type=int)

    args = parser.parse_args()
    nheads = [int(item) for item in args.nheads.split(',')]
    
    main(args.dataset, 
        nheads, args.hidden_units, args.feat_drop_rate, args.coefs_drop_rate,
        args.epochs, args.learning_rate, args.l2_weight, args.patience, 
        args.data_seed, args.net_seed, args.checkpoint_path, args.verbose)
