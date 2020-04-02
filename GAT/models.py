import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from GAT.layers import Attention_layer


class GAT(tf.keras.Model):

    def __init__(self, graph, nlabels, nhidden, nheads, drop_rate):
        super().__init__()
        
        self.att1 = Attention_layer(graph, nhidden, 'elu', nheads=nheads[0], reduction=False)
        self.att2 = Attention_layer(graph, nlabels, 'softmax', nheads=nheads[1], reduction=True)

        self.drop_rate = drop_rate

    def call(self, inputs, training=True):

        drop_rate = 1.0-self.drop_rate if training else 0.0

        drop_inp = tf.nn.dropout(inputs, drop_rate)
        h_features = self.att1(drop_inp)

        drop_h = tf.nn.dropout(h_features, drop_rate)
        out = self.att2(drop_h)

        return tf.squeeze(out)
