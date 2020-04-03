import tensorflow as tf
import numpy as np

from time import time 

class Attention_layer(tf.keras.layers.Layer):

    def __init__(self, graph, output_size, activation, nheads, coefs_drop_rate, reduction, **kwargs):
        # if nheads==1 Single head attention layer, otherwise Multi head attention layer
        super(Attention_layer, self).__init__(**kwargs)
        self.nheads = nheads
        self.graph = graph
        self.output_size = output_size
        self.coefs_drop_rate = coefs_drop_rate
        self.reduction = reduction
        self.non_linearity = tf.keras.activations.get(activation)


    def build(self, input_shape):

        self.Ws = self.add_weight(shape=[self.nheads, self.output_size, input_shape[1]], 
                                initializer='glorot_uniform', name='Ws', trainable=True)

        self.As = self.add_weight(shape=[self.nheads, 2*self.output_size, 1],
                                initializer='glorot_uniform', name='As', trainable = True)

        super().build(input_shape)


    def call(self, inputs):

        graph = self.toSparseTensor()
        As = self.As
        Ws = self.Ws
        nheads = self.nheads
        out_size = self.output_size

        t_nodes = tf.matmul(inputs, tf.transpose(Ws, [0,2,1])) 
        Al = tf.matmul(t_nodes, As[:, :out_size, :])
        Ar = tf.matmul(t_nodes, As[:, out_size:, :])

        for k in range(self.nheads):

            row_partial = graph * Al[k]
            col_partial = graph * Ar[k]
            E = tf.sparse.add(row_partial, tf.sparse.transpose(col_partial))
            # Sparse LReLU
            lrelu = tf.SparseTensor(E.indices, tf.nn.leaky_relu(E.values), E.dense_shape)
            
            alphas = tf.sparse.softmax(E)

            # Dropout on attention coefficients
            if self.coefs_drop_rate != 0.0:
                alphas = tf.SparseTensor(E.indices, tf.nn.dropout(E.values,0.4), E.dense_shape)

            h_k_out = tf.sparse.sparse_dense_matmul(alphas, t_nodes[k])

            if k == 0:
                out = tf.expand_dims(h_k_out, 0)
            else: 
                out = tf.concat((out,tf.expand_dims(h_k_out, 0)), axis=0)

    
        out = tf.transpose(out, [1,0,2])

        if self.reduction:
            out = tf.reduce_mean(out, axis=1, keepdims=True)
        else:
            out = tf.reshape(out, [out.shape[0],-1])

        return out if self.non_linearity is None else self.non_linearity(out) 

    def toSparseTensor(self):
        coo_graph = self.graph.tocoo()
        indices = np.mat([coo_graph.row, coo_graph.col]).transpose()
        return tf.cast( tf.sparse.SparseTensor(indices, coo_graph.data, coo_graph.shape), self._dtype )