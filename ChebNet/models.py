import tensorflow as tf
import numpy as np
from add_parent_path import add_parent_path

from layers import Chebychev

with add_parent_path():
    from metrics import *

class ChebNet(tf.keras.models.Sequential):

    def __init__(self, norm_L=None, K=3, num_classes=2, dropout_rate=None, hidden_units=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dropout_rate is not None: self.add(tf.keras.layers.Dropout(dropout_rate))
        if norm_L       is not None: self.add(Chebychev(norm_L, K, num_filters=hidden_units, activation="relu"))
        if dropout_rate is not None: self.add(tf.keras.layers.Dropout(dropout_rate))
        if norm_L       is not None: self.add(Chebychev(norm_L, K, num_filters=num_classes, activation="softmax"))
