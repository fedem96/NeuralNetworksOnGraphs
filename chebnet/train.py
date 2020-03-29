import tensorflow as tf
from layers import Chebychev

import sys, os
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import *

if __name__ == "__main__":
    
    K = 3
    dataset_name = "cora"

    # read dataset
    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    # shuffle dataset
    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    # get masks
    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, len(labels))

    print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)
    print("calculating laplacian matrix")
    norm_L = normalized_laplacian_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    print("defining model")
    # input_shape: (num_nodes, features)
    model = tf.keras.Sequential([
        Chebychev(norm_L, K, num_filters=8, activation="relu"),
        Chebychev(norm_L, K, num_filters=16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    preds = model.predict(features, batch_size=num_nodes)
    print("preds")
    print(preds)
