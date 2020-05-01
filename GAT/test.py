import argparse
import numpy as np
import scipy.sparse as sp

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from add_parent_path import add_parent_path

from models import GAT

with add_parent_path():
    from metrics import masked_accuracy, masked_loss
    from utils import *

def main(dataset_name, yang_splits,
        nheads, hidden_units, feat_drop_rate, 
        coefs_drop_rate, l2_weight,
        data_seed, checkpoint_path, verbose):

    # reproducibility
    np.random.seed(data_seed)

    if yang_splits:
        features, o_h_labels, graph, mask_train, mask_val, mask_test = read_dataset(dataset_name, yang_splits=True)
        labels = np.array([np.argmax(l) for l in o_h_labels], dtype=np.int32)
    else:
        if verbose > 0: print("reading dataset")
        features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
        num_classes = len(set(labels))

        if verbose > 0: print("shuffling dataset")
        features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
        
        if verbose > 0: print("obtaining masks")
        mask_train, mask_val, mask_test = split(dataset_name, labels)

        if verbose > 0: print("calculating adjacency matrix")
        graph = adjacency_matrix(neighbors)

    # add self loops to adj matrix
    graph = graph + sp.eye(graph.shape[0])
    num_classes = get_num_classes(dataset_name)
    features = normalize_features(features)
    
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    if verbose > 0: print("defining model")
    model = GAT(graph, num_classes, hidden_units, nheads, feat_drop_rate, coefs_drop_rate)

    model.compile(loss=lambda y_true, y_pred: masked_loss(y_true, y_pred)+l2_weight*tf.reduce_sum([tf.nn.l2_loss(w) for w in model.weights if not 'bias' in w.name]), 
                    metrics=[masked_accuracy])

    print("load model from checkpoint")
    wpath = os.path.join(checkpoint_path,'cp.ckpt')
    model.load_weights(wpath).expect_partial()

    if verbose > 0: print("test the model on test set")
    t_loss, t_accuracy = model.evaluate(features, y_test, batch_size=len(features), verbose=0)
    print("accuracy on test: " + str(t_accuracy))

    intermediate_output = model.call(features, training=False, intermediate=True)
    plot_tsne(intermediate_output[mask_test], labels[mask_test], len(o_h_labels[0]), 'GAT')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GAT')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-y", "--yang-splits", help="whether to use Yang splits or not", default=True, action='store_true')
    
    # network hyperparameters
    parser.add_argument('-nh', '--nheads', help='heads number per layer (the len of the list represent the model layers number)', default='8,1')
    parser.add_argument("-hu", "--hidden-units", help="number of Graph Convolutional filters in the first layer", default=8, type=int)
    parser.add_argument("-fd", "--feat-drop-rate", help="dropout rate for model dropout layers (fraction of the input units to drop)", default=0.4, type=float)
    parser.add_argument("-cd", "--coefs-drop-rate", help="dropout rate for attention coefficients (fraction of the input units to drop)", default=0.4, type=float)

    # optimization parameters
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)
 
    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for model checkpoints")

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    args = parser.parse_args()
    nheads = [int(item) for item in args.nheads.split(',')]
    
    main(args.dataset, args.yang_splits,
        nheads, args.hidden_units, args.feat_drop_rate, 
        args.coefs_drop_rate, args.l2_weight,
        args.data_seed, args.checkpoint_path, args.verbose)
