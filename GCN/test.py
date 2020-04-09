import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import GCN
from layers import GraphConvolution

with add_parent_path():
    from metrics import masked_accuracy, masked_loss
    from utils import *

def main(dataset_name,
        dropout_rate, hidden_units,
        training_epochs, learning_rate, l2_weight,
        data_seed, net_seed,
        model_path):

    # reproducibility
    np.random.seed(data_seed)

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
    A = adjacency_matrix(neighbors)
    print("calculating renormalized matrix")
    renormalized_matrix = renormalization_matrix(A)

    print("load model from checkpoint")
    #model = tf.keras.models.load_model(model_path, custom_objects={'GCN':GCN, 'GraphConvolution': GraphConvolution}, compile=False)
    model = GCN(renormalized_matrix, num_classes, dropout_rate, hidden_units)
    model.compile(
        loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(model.trainable_weights[0]), # regularize first layer only
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[masked_accuracy],
        run_eagerly=True
    )
    model.build(features.shape)
    model.load_weights(model_path)

    print("test the model on test set")
    loss, accuracy = model.evaluate(features, y_test, batch_size=len(features), verbose=0)
    print("accuracy on test: " + str(accuracy))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GCN')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])

    # network hyperparameters
    parser.add_argument("-dr", "--dropout-rate", help="dropout rate for dropout layers (fraction of the input units to drop)", default=0.5, type=float)
    parser.add_argument("-hu", "--hidden-units", help="number of Graph Convolutional filters in the first layer", default=16, type=int)

    # optimization hyperparameters
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=20, type=int)
    parser.add_argument("-lr", "--learning-rate", help="starting learning rate of Adam optimizer", default=0.01, type=float)
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model to file
    parser.add_argument("-m", "--model", help="path where to save model", default=None)

    args = parser.parse_args()

    if args.model is None:
        print("Error: model to test not specified")
        exit(1)

    main(args.dataset,
        args.dropout_rate, args.hidden_units,
        args.epochs, args.learning_rate, args.l2_weight,
        args.data_seed, args.net_seed,
        args.model)
