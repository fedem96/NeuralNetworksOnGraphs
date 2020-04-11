import numpy as np
import tensorflow as tf

from layers import GraphConvolution


class GCN(tf.keras.models.Sequential):

    def __init__(self, renormalized_matrix=None, num_classes=2, dropout_rate=None, hidden_units=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dropout_rate        is not None: self.add(tf.keras.layers.Dropout(dropout_rate))
        if renormalized_matrix is not None: self.add(GraphConvolution(renormalized_matrix, num_filters=hidden_units, activation="relu"))
        if dropout_rate        is not None: self.add(tf.keras.layers.Dropout(dropout_rate))                                                
        if renormalized_matrix is not None: self.add(GraphConvolution(renormalized_matrix, num_filters=num_classes, activation="softmax"))
