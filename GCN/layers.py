import importlib

import numpy as np
import tensorflow as tf

# utils = importlib.import_module('..utils', '.')
# from utils import *

class GraphConvolution(tf.keras.layers.Layer):

    # Graph Convolutional Layer
    # renormalized_matrix = renorm_D_minus_half * renorm_A * renorm_D_minus_half
    # renorm_D_minus_half: renormalized 
    # renorm_A:
    # computes: renormalized_matrix * X * theta

    def __init__(self, renormalized_matrix, num_filters, activation):
        super().__init__()

        self._dtype = tf.float32

        coo_mat = renormalized_matrix.tocoo()
        indices = np.mat([coo_mat.row, coo_mat.col]).transpose()
        self.renormalized_matrix = tf.cast(tf.sparse.SparseTensor(indices, coo_mat.data, coo_mat.shape), self._dtype)

        self.n = self.renormalized_matrix.shape[0] # number of nodes
        self.fin = -1                              # number of input features
        self.fout = num_filters                    # number of output features
        
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.fin = input_shape[1]
        self.theta = self.add_weight(shape=[self.fin, self.fout], initializer='glorot_uniform', dtype=self._dtype)

    def call(self, x):
        x = tf.cast(x, self._dtype)
        mx = tf.sparse.sparse_dense_matmul(self.renormalized_matrix, x) # shapes: (n, n)   *   (n, fin)  -> (n, fin)
        o = tf.matmul(mx, self.theta)                                   # shapes: (n, fin) * (fin, fout) -> (n, fout)
        return o if self.activation is None else self.activation(o)
