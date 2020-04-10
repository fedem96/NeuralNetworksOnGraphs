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

    def __init__(self, renormalized_matrix=None, num_filters=16, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if renormalized_matrix is not None:
            coo_mat = renormalized_matrix.tocoo()
            self.coo_mat_indices = np.mat([coo_mat.row, coo_mat.col]).transpose()
            self.coo_mat_data = coo_mat.data
            self.coo_mat_shape = coo_mat.shape

        #self.n = renormalized_matrix.shape[0] # number of nodes
        self.fout = num_filters                # number of output features
        
        if activation is not None: self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        fin = input_shape[1]
        self.theta = self.add_weight(shape=[fin, self.fout], initializer='glorot_uniform', dtype=tf.float32, name="theta")
        self.bias = self.add_weight(shape=[self.fout], initializer='zeros', dtype=tf.float32, name="bias")
        self.renormalized_matrix = tf.cast(tf.sparse.SparseTensor(self.coo_mat_indices, self.coo_mat_data, self.coo_mat_shape), tf.float32)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        mx = tf.sparse.sparse_dense_matmul(self.renormalized_matrix, x) # shapes: (n, n)   *   (n, fin)  -> (n, fin)
        o = tf.matmul(mx, self.theta)                                   # shapes: (n, fin) * (fin, fout) -> (n, fout)
        o = tf.nn.bias_add(o, self.bias)
        return o if self.activation is None else self.activation(o)
    
    # def get_config(self):
    #     base_config = super().get_config()
    #     config = {
    #         'coo_mat_indices': self.coo_mat_indices,
    #         'coo_mat_data': self.coo_mat_data,
    #         'coo_mat_shape': self.coo_mat_shape,
    #         'fout': self.fout,
    #     }
    #     return dict(list(base_config.items()) + list(config.items()))
