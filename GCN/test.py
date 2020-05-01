import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import GCN
from layers import GraphConvolution

with add_parent_path():
    from metrics import masked_accuracy, masked_loss
    from utils import *

def main(dataset_name, yang_splits,
        dropout_rate, hidden_units,
        l2_weight,
        data_seed,
        model_path, verbose,
        tsne):

    # reproducibility
    np.random.seed(data_seed)

    if yang_splits:
        features, o_h_labels, A, mask_train, mask_val, mask_test = read_dataset(dataset_name, yang_splits=True)
        labels = np.array([np.argmax(l) for l in o_h_labels], dtype=np.int32)
    else:
        if verbose > 0: print("reading dataset")
        features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)

        if verbose > 0: print("shuffling dataset")
        features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
        
        if verbose > 0: print("obtaining masks")
        mask_train, mask_val, mask_test = split(dataset_name, labels)

        if verbose > 0: print("calculating adjacency matrix")
        A = adjacency_matrix(neighbors)

    num_classes = get_num_classes(dataset_name)
    features = normalize_features(features)

    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    if verbose > 0: print("calculating renormalized matrix")
    renormalized_matrix = renormalization_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    if verbose > 0: print("load model from checkpoint")
    model = GCN(renormalized_matrix, num_classes, dropout_rate, hidden_units)
    model.compile(
        loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(model.trainable_weights[0]), # regularize first layer only
        #optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[masked_accuracy],
        #run_eagerly=True
    )
    model.build(features.shape)
    model.summary()
    model.load_weights(os.path.join(model_path, "ckpt")).expect_partial()

    if verbose > 0: print("test the model on test set")
    loss, accuracy = model.evaluate(features, y_test, batch_size=num_nodes, verbose=0)
    print("accuracy on test: " + str(accuracy))

    if tsne:
        if verbose > 0: print("calculating t-SNE plot")
        intermediate_layer_model = tf.keras.Sequential([model.layers[0], model.layers[1]])
        intermediate_output = intermediate_layer_model.predict(features, batch_size=num_nodes)
        plot_tsne(intermediate_output[mask_test], labels[mask_test], len(o_h_labels[0]), 'GCN')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GCN')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-y", "--yang-splits", help="whether to use Yang splits or not", default=False, action='store_true')

    # network hyperparameters
    parser.add_argument("-dr", "--dropout-rate", help="dropout rate for dropout layers (fraction of the input units to drop)", default=0.5, type=float)
    parser.add_argument("-hu", "--hidden-units", help="number of Graph Convolutional filters in the first layer", default=16, type=int)

    # # optimization hyperparameters
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)

    # save model to file
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    # tsne
    parser.add_argument("-t", "--tsne", help="whether to make t-SNE plot or not", default=False, action='store_true')

    args = parser.parse_args()

    if args.checkpoint_path is None:
        print("Error: model to test not specified")
        exit(1)

    main(args.dataset, args.yang_splits,
        args.dropout_rate, args.hidden_units,
        args.l2_weight,
        args.data_seed,
        args.checkpoint_path, args.verbose,
        args.tsne)
