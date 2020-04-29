import tensorflow as tf
import numpy as np

from layers import Attention_layer

class GAT(tf.keras.Model):

    def __init__(self, graph, nlabels, nhidden, nheads, feat_drop_rate, coefs_drop_rate):
        super().__init__()
        
        self.atts = [Attention_layer(graph, nhidden, 'elu', heads, feat_drop_rate, coefs_drop_rate, reduction=False)
                    for heads in nheads[:-1]]
        self.att_out = Attention_layer(graph, nlabels, 'softmax', nheads[-1], feat_drop_rate, coefs_drop_rate, reduction=True)

        self.feat_drop_rate = feat_drop_rate

        
    def call(self, x, training=True, intermediate=False):
        # Input x has shape (n_nodes, features)
        feat_dr = self.feat_drop_rate if training else 0.0

        for att in self.atts:
            drop_inp = tf.nn.dropout(x, feat_dr)
            x = att(drop_inp, training)
        
        if intermediate:
            return x

        drop_h = tf.nn.dropout(x, feat_dr)
        out = self.att_out(drop_h, training)

        return tf.squeeze(out)
