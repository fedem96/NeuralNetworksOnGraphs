import sys, os

import numpy as np
import tensorflow as tf
from add_parent_path import add_parent_path

from layers import GraphConvolution

with add_parent_path():
    from metrics import *


class GCN(tf.keras.models.Sequential):

    def __init__(self, renormalized_matrix, num_classes, dropout_rate, hidden_units, learning_rate, l2_weight):
        super().__init__([
            tf.keras.layers.Dropout(dropout_rate),
            GraphConvolution(renormalized_matrix, num_filters=hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            GraphConvolution(renormalized_matrix, num_filters=num_classes, activation="softmax"),
        ])

        self.compile(
            loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(self.trainable_weights[0]), # regularize first layer only
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[masked_accuracy]
        )
