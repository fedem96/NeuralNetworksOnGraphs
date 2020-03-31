import tensorflow as tf
import numpy as np


class Attention_layer(tf.keras.layers.Layer):

    def __init__(self, neighbors, output_size, activation, nheads, **kwargs):
        # if nheads==1 Single head attention layer, otherwise Multi head attention layer
        super(Attention_layer, self).__init__(**kwargs)
        self.nheads = nheads
        self.neighbors = neighbors
        self.activation = activation
        self.output_size = output_size
        self.drop_prob = 0.6  # prob for dropout before

        self.lrelu = tf.keras.layers.LeakyReLU(0.2)
        self.non_linearity = tf.keras.activations.get(self.activation)

    def build(self, input_shape):

        self.Ws = self.add_weight(shape=[self.nheads, self.output_size, input_shape[1]], 
                                initializer='glorot_uniform', name='Ws', trainable=True)

        self.As = self.add_weight(shape=[self.nheads, 2*self.output_size, 1],
                                initializer='glorot_uniform', name='As', trainable = True)

        super().build(input_shape)

    def call(self, inputs):

        # out = tf.zeros((self.nheads, inputs.shape[0], self.output_size), dtype=tf.float32)
        # t_nodes = tf.matmul(self.Ws, tf.expand_dims(inputs,-1))

        for k in range(self.nheads):
            
            # W trasformations (with also the node itself)
            t_nodes = tf.matmul(self.Ws[k], tf.expand_dims(inputs,-1))
            for n in range(len(t_nodes)):
                # att_coef = []
                # t_neigh = np.zeros((len(self.neighbors[n])+1, t_nodes.shape[1], t_nodes.shape[2]), dtype=np.float32)
                t_neigh = tf.transpose(t_nodes[n])
                conc_node = tf.concat((t_nodes[n],t_nodes[n]),axis=0)
                c = tf.matmul(tf.transpose(self.As[k]), conc_node)
                att_coef = tf.expand_dims(tf.exp(self.lrelu(tf.squeeze(c))),0)
                for i, v in enumerate(self.neighbors[n]):
                    t_neigh = tf.concat((t_neigh,tf.transpose(t_nodes[v[0]])), axis=0)
                    conc_node = tf.concat((t_nodes[n],t_nodes[v[0]]),axis=0)
                    exp_n = tf.exp(self.lrelu(tf.squeeze(tf.matmul(tf.transpose(self.As[k]), conc_node))))
                    att_coef = tf.concat((att_coef, tf.expand_dims(exp_n,0)), axis=0)
                alphas = tf.nn.softmax(att_coef)
                b = tf.multiply(tf.expand_dims(alphas,-1), tf.squeeze(t_neigh))
                if n == 0:
                    node_out = tf.reduce_sum(b, axis=0, keepdims=True)
                else:
                    node_out = tf.concat((node_out,  tf.reduce_sum(b, axis=0, keepdims=True)), axis=0)

            if k == 0:
                out = tf.expand_dims(node_out, 0)
            else: 
                out = tf.concat((out,tf.expand_dims(node_out, 0)), axis=0)
        
        out = tf.transpose(out, perm=[1,0,2])

        if self.activation == 'softmax' or self.activation == 'sigmoid':
            out = tf.reduce_mean(out, axis=1, keepdims=True)
        else:
            out = tf.reshape(out, [out.shape[0],-1])

        return self.non_linearity(out)


class GAT(tf.keras.Model):

    def __init__(self, neighbors, nlabels, n_output_heads):
        super().__init__()

        self.drop1 = tf.keras.layers.Dropout(0.6)
        self.att1 = Attention_layer(neighbors, 8, 'elu', nheads=2)
        self.drop2 = tf.keras.layers.Dropout(0.6)
        self.att2 = Attention_layer(neighbors, nlabels, 'softmax', nheads=n_output_heads)

    def call(self, inputs):

        drop_inp = self.drop1(inputs)
        h_features = self.att1(drop_inp)

        drop_h = self.drop2(h_features)
        out = self.att2(drop_h)

        return tf.squeeze(out)
