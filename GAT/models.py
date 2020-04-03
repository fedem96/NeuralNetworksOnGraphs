import tensorflow as tf
import numpy as np

from layers import Attention_layer

class GAT(tf.keras.Model):

    def __init__(self, graph, nlabels, nhidden, nheads, feat_drop_rate, coefs_drop_rate):
        super().__init__()
        
        self.att1 = Attention_layer(graph, nhidden, 'elu', nheads[0], coefs_drop_rate, reduction=False)
        self.att2 = Attention_layer(graph, nlabels, 'softmax', nheads[1], coefs_drop_rate, reduction=True)

        self.feat_drop_rate = feat_drop_rate

        
    def call(self, inputs, training=True):

        drop_rate = self.feat_drop_rate if training else 0.0

        drop_inp = tf.nn.dropout(inputs, drop_rate)
        h_features = self.att1(drop_inp)

        drop_h = tf.nn.dropout(h_features, drop_rate)
        out = self.att2(drop_h)

        return tf.squeeze(out)
