import tensorflow as tf
import argparse

from layers import Chebychev

import sys, os
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import *


def eval_accuracy(model, X, labels, mask):
    preds = model.predict(X, batch_size=len(X))
    pred_classes = np.argmax(preds[mask], axis=1)
    return sum([p==l for p,l in zip(pred_classes, labels)]) / np.sum(mask)


def main(dataset_name, training_epochs):
    K = 3

    data_seed = 0
    # net_seed = 0 # TODO: implement

    # read dataset
    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    # shuffle dataset
    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys, data_seed)
    # get masks
    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, len(labels))
    X_train = np.multiply(features, np.broadcast_to(mask_train.T, features.T.shape).T )
    X_val   = np.multiply(features, np.broadcast_to(mask_val.T,   features.T.shape).T )
    X_test  = np.multiply(features, np.broadcast_to(mask_test.T,  features.T.shape).T )

    print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)
    print("calculating laplacian matrix")
    norm_L = normalized_laplacian_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    print("defining model")
    # input_shape: (num_nodes, features)
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.5),
        Chebychev(norm_L, K, num_filters=32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        Chebychev(norm_L, K, num_filters=32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        regularizer=tf.keras.regularizers.l2(5e-4),
        metrics=['accuracy']
    )

    # Convert labels to categorical one-hot encoding
    model.fit(X_train, labels, epochs=training_epochs, batch_size=num_nodes)

    print("validation accuracy", eval_accuracy(model, X_val, labels, mask_val))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a chebnet')
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=100, type=int)
    args = parser.parse_args()

    main(args.dataset, args.epochs)