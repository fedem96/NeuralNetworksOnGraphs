import tensorflow as tf
from layers import Chebychev

import sys, os
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import *

if __name__ == "__main__":
    
    max_order = 2
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

    num_nodes = len(A)
    num_features = len(features[0])

    print("defining model")
    # input_shape: (num_nodes, features)
    model = tf.keras.Sequential([
        Chebychev(laplacian=norm_L, max_order=max_order, num_filters=1),
        # Chebychev(laplacian=norm_L, max_order=max_order, num_filters=8),
        # tf.keras.layers.Dense(8),
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    x = tf.random.normal([num_nodes, 1])
    print("X:", x)
    preds = model.predict(x, batch_size=len(x))
    print(preds)
