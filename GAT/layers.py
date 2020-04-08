import tensorflow as tf
import numpy as np

from time import time 

class Attention_layer(tf.keras.layers.Layer):

    def __init__(self, graph, output_size, activation, nheads, feat_drop_rate, coefs_drop_rate, reduction, **kwargs):
        # if nheads==1 Single head attention layer, otherwise Multi head attention layer
        super(Attention_layer, self).__init__(**kwargs)
        self.nheads = nheads
        self.graph = graph
        self.output_size = output_size
        self.coefs_drop_rate = coefs_drop_rate  
        self.feat_drop_rate = feat_drop_rate
        self.reduction = reduction
        self.non_linearity = tf.keras.activations.get(activation)


    def build(self, input_shape):

        self.Ws = self.add_weight(shape=[self.nheads, self.output_size, input_shape[1]], 
                                initializer='glorot_uniform', name='Ws', trainable=True)

        self.Ws_bias = self.add_weight(shape=[self.nheads, self.output_size], initializer='zeros', 
                                    trainable = True, dtype=self._dtype)


        self.As = self.add_weight(shape=[self.nheads, 2*self.output_size, 1],
                                initializer='glorot_uniform', name='As', trainable = True)
        
        self.As_bias = self.add_weight(shape=[self.nheads, 2], initializer='zeros',  # 8 x 1
                                    trainable = True, dtype=self._dtype)

        super().build(input_shape)


    def call(self, inputs, training):

        graph = self.toSparseTensor()
        coefs_dr = self.coefs_drop_rate if training else 0.0
        feat_dr = self.feat_drop_rate if training else 0.0

        t_nodes = tf.matmul(inputs, tf.transpose(self.Ws, [0,2,1])) 
        Al = tf.matmul(t_nodes, self.As[:, :self.output_size, :])
        Ar = tf.matmul(t_nodes, self.As[:, self.output_size:, :])

        for k in range(self.nheads):
            
            row_partial = graph * (Al[k]+self.As_bias[k,0]) 
            col_partial = graph * (Ar[k]+self.As_bias[k,1]) 
            E = tf.sparse.add(row_partial, tf.sparse.transpose(col_partial))
            # Sparse LReLU
            lrelu = tf.SparseTensor(E.indices, tf.nn.leaky_relu(E.values), E.dense_shape)
            
            alphas = tf.sparse.softmax(E)

            # Dropout on attention coefficients
            alphas = tf.SparseTensor(alphas.indices, tf.nn.dropout(alphas.values,coefs_dr), alphas.dense_shape)
            
            # Dropout on input features
            h_k_out = tf.sparse.sparse_dense_matmul(alphas, tf.nn.dropout(t_nodes[k],feat_dr))
            
            # Add bias
            h_k_out = tf.nn.bias_add(h_k_out, self.Ws_bias[k])

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