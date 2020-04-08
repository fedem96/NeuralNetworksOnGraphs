import tensorflow as tf
import numpy as np
from add_parent_path import add_parent_path

from layers import Chebychev

with add_parent_path():
    from metrics import *

class ChebNet(tf.keras.models.Sequential):

    def __init__(self, norm_L, K, num_classes, dropout_rate, hidden_units, learning_rate, l2_weight):
        super().__init__([
            tf.keras.layers.Dropout(dropout_rate),
            Chebychev(norm_L, K, num_filters=hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            Chebychev(norm_L, K, num_filters=num_classes, activation="softmax"),
        ])

        self.compile(
            loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(self.trainable_weights[0]), # regularize first layer only
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[masked_accuracy],
            # run_eagerly=True
        )
