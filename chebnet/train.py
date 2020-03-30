import tensorflow as tf
import argparse

from models import ChebNet

import sys, os
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import *


def eval_accuracy(model, X, y, mask):
    preds = model.predict(X, batch_size=len(X))
    pred_classes = np.argmax(preds[mask], axis=1)
    true_classes = np.argmax(y[mask], axis=1)
    return np.sum(pred_classes == true_classes) / np.sum(mask)


def main(dataset_name, training_epochs):
    K = 4

    data_seed = 0
    net_seed = 0
    tf.random.set_seed(net_seed)

    # read dataset
    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))
    # shuffle dataset
    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys, data_seed)
    # get masks
    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)
    X_train = np.multiply(features, np.broadcast_to(mask_train.T, features.T.shape).T )
    X_val   = np.multiply(features, np.broadcast_to(mask_val.T,   features.T.shape).T )
    X_test  = np.multiply(features, np.broadcast_to(mask_test.T,  features.T.shape).T )
    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)
    print("calculating laplacian matrix")
    norm_L = normalized_laplacian_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    print("defining model")
    # input_shape: (num_nodes, features)
    model = ChebNet(norm_L, K, num_classes)

    # Convert labels to categorical one-hot encoding
    #model.fit(features, y_train, epochs=training_epochs, batch_size=num_nodes, shuffle=False)
    model.train(X_train, y_train, epochs=training_epochs)

    print("validation accuracy", eval_accuracy(model, X_val, y_val, mask_val))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a chebnet')
    parser.add_argument("-d", "--dataset", help="dataset to use", default="citeseer", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10, type=int)
    args = parser.parse_args()

    main(args.dataset, args.epochs)