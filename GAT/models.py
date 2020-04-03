import tensorflow as tf
import numpy as np
from add_parent_path import add_parent_path

from layers import Attention_layer

with add_parent_path():
    from metrics import masked_accuracy, masked_loss


class GAT(tf.keras.Model):

    def __init__(self, graph, nlabels, nhidden, nheads, feat_drop_rate, 
                coefs_drop_rate, l2_weight, optimizer):
        super().__init__()
        
        self.att1 = Attention_layer(graph, nhidden, 'elu', nheads[0], coefs_drop_rate, reduction=False)
        self.att2 = Attention_layer(graph, nlabels, 'softmax', nheads[1], coefs_drop_rate, reduction=True)

        self.feat_drop_rate = feat_drop_rate

        self.compile(loss=lambda y_true, y_pred: masked_loss(y_true, y_pred) + l2_weight * tf.nn.l2_loss(y_pred-y_true), 
                    optimizer=optimizer, metrics=[masked_accuracy])

    def call(self, inputs, training=True):

        drop_rate = self.feat_drop_rate if training else 0.0

        drop_inp = tf.nn.dropout(inputs, drop_rate)
        h_features = self.att1(drop_inp)

        drop_h = tf.nn.dropout(h_features, drop_rate)
        out = self.att2(drop_h)

        return tf.squeeze(out)
