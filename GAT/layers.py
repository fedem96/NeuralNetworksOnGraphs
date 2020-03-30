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
            # MEAN IF ACTIVATION IS SOFTMAX OR SIGMOID AKA LAST LAYER
            out = tf.reduce_mean(out, axis=1, keepdims=True)
        else:
            out = tf.reshape(out, [out.shape[0],-1])

        return self.non_linearity(out)


class GAT(tf.keras.Model):

    def __init__(self, neighbors, nlabels, n_output_heads):
        super().__init__()

        self.drop1 = tf.keras.layers.Dropout(0.6)
        self.att1 = Attention_layer(neighbors, 8, 'elu', nheads=8)
        self.drop2 = tf.keras.layers.Dropout(0.6)
        self.att2 = Attention_layer(neighbors, nlabels, 'softmax', nheads=n_output_heads)

    def call(self, inputs):

        # drop_inp = self.drop1(inputs)
        h_features = self.att1(inputs)

        # drop_h = self.drop2(h_features)
        out = self.att2(h_features)

        return tf.squeeze(out)





''' 
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
        # FIXME: Non tutti i nodi ma solo quelli di train?? da vedere!!!!! secondo me si con accesos pero a tutti i nodi!!!!
        out = []
        # for each node in inputs, we compute its attention coefs
        # with its neighbors (and with itself)
        for n in range(inputs.shape[0]):
            # att_coefs, t_nodes = self.attention_coef(n, inputs)

            #FIXME: SI ROMPE
            t_nodes = []
            conc_nodes = []

            # W trasformations (with also the node itself)
            t_nodes.append(tf.matmul(self.Ws, tf.reshape(inputs[n], [-1, 1])))
            conc_nodes.append(tf.concat((t_nodes[-1], t_nodes[-1]), axis=1))
            for j in self.neighbors[n]:
                t_nodes.append(tf.matmul(self.Ws, tf.reshape(inputs[j[0]], [-1, 1])))
                conc_nodes.append(tf.concat((t_nodes[0], t_nodes[-1]), axis=1))
            
            t_nodes = tf.Variable(t_nodes)
            conc_nodes = tf.Variable(conc_nodes)#tf.transpose(,perm=[1,0,2,3])

            
            den = np.full((len(self.neighbors[n])+1, self.nheads), 1e-7, dtype=np.float32)  # for stability
            att_coef = np.zeros_like(den)
            for k in range(len(self.neighbors[n])+1):
                den[k] = tf.squeeze(tf.exp(self.lrelu(tf.matmul(tf.transpose(self.As, perm=[0, 2, 1]), conc_nodes[k]))))
            den = tf.reduce_sum(den, axis=0)

            for idx in range(len(self.neighbors[n])+1):
                att_coef[idx] = tf.divide(tf.squeeze(tf.exp(self.lrelu(tf.matmul(tf.transpose(self.As, perm=[0, 2, 1]), conc_nodes[idx])))), den)

            
            att_coefs = tf.transpose(att_coef)
            t_nodes = tf.transpose(tf.reshape(t_nodes, [len(self.neighbors[n])+1, self.nheads, -1]), perm=[1,0,2])


            node_sums = tf.reduce_sum(tf.multiply(tf.reshape(att_coefs, [att_coefs.shape[0], att_coefs.shape[1],-1]), t_nodes), axis=1)

            if self.activation == 'softmax' or self.activation == 'sigmoid':
                # MEAN IF ACTIVATION IS SOFTMAX OR SIGMOID AKA LAST LAYER
                node_sums = tf.reduce_mean(node_sums, axis=0, keepdims=True)
            else:
                node_sums = tf.reshape(node_sums, [-1,1])

            out.append(self.non_linearity(node_sums))

        return tf.Variable(tf.squeeze(out))

    # @tf.function
    def attention_coef(self, node, inputs):
        # TODO: DOVE METTO IL DROPOUT??
        # store vecotrs after the W transformations
        t_nodes = []
        conc_nodes = []

        # W trasformations (with also the node itself)
        t_nodes.append(tf.matmul(self.Ws, tf.reshape(inputs[node], [-1, 1])))
        conc_nodes.append(tf.concat((t_nodes[-1], t_nodes[-1]), axis=1))
        for j in self.neighbors[node]:
            t_nodes.append(tf.matmul(self.Ws, tf.reshape(inputs[j[0]], [-1, 1])))
            conc_nodes.append(tf.concat((t_nodes[0], t_nodes[-1]), axis=1))
        
        t_nodes = tf.Variable(t_nodes)
        conc_nodes = tf.Variable(conc_nodes)#tf.transpose(,perm=[1,0,2,3])

        
        den = np.full((len(self.neighbors[node])+1, self.nheads), 1e-7, dtype=np.float32)  # for stability
        att_coef = np.zeros_like(den)
        for k in range(len(self.neighbors[node])+1):
            den[k] = tf.squeeze(tf.exp(self.lrelu(tf.matmul(tf.transpose(self.As, perm=[0, 2, 1]), conc_nodes[k]))))
        den = tf.reduce_sum(den, axis=0)

        for idx in range(len(self.neighbors[node])+1):
            att_coef[idx] = tf.divide(tf.squeeze(tf.exp(self.lrelu(tf.matmul(tf.transpose(self.As, perm=[0, 2, 1]), conc_nodes[idx])))), den)

        return tf.transpose(att_coef), tf.transpose(tf.reshape(t_nodes, [len(self.neighbors[node])+1, self.nheads, -1]), perm=[1,0,2])
 '''





""" FIXME: DOVREBBERO ESSERE INUTILI
class Attention_coefficient(tf.keras.layers.Layer):

    def __init__(self, W, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.W = W

        a_init = tf.keras.initializers.GlorotUniform()
        self.a = tf.Variable(initial_value=a_init(shape=(2*output_dim,1), dtype=tf.float32), trianable=True)
        self.lrelu = tf.keras.layers.LeakyReLU(0.2)

        def call(self, input, features, neighbors):
            for n in range(len(features)):
                a_n = self.attention_coef(n, features, neighbors) 


        def attention_coef(self, node, features, neighbors):
    
            i_tras = tf.matmul(W, features[node])
            i_conc = tf.concat((i_tras,i_tras), axis=0)
            
            h_j = tf.Variable((len(neighbors[node]), 2*len(features[node])), dtype=tf.float32)
            
            for idx, j in enumerate(neighbors[node]):
                j_tras = tf.matmul(W, features[j])
                h_j[idx] = tf.concat((i_tras, j_tras), axis=0)

            return h_j
 """
""" 

class Single_Head_Attention(tf.keras.layers.Layer):

    def __init__(self, features, neighbors, output_size, activation):
        super().__init__()
        self.features = features
        self.neighbors = neighbors

        # self.W = Linear(output_size)
        w_init = tf.keras.initializers.GlorotUniform()
        self.W = tf.Variable(initial_value=w_init(output_size, len(
            self.features[0]), dtype=tf.float32), trainable=True)

        a_init = tf.keras.initializers.GlorotUniform()
        self.a = tf.Variable(initial_value=a_init(
            shape=(2*output_size, 1), dtype=tf.float32), trianable=True)

        self.lrelu = tf.keras.layers.LeakyReLU(0.2)

        self.non_linearity = tf.keras.activations.get(activation)

    def call(self, inputs):
        # inputs are the list of some nodes # FIXME: (mini batch?)
        out = tf.Variable((len(inputs), 1), trainable=True)
        # for each node in inputs, we compute its attention coefs
        # with its neighbors (and with itself)
        for n in range(len(inputs)):
            att_coefs, t_nodes = self.attention_coef(n)
            out[n] = self.non_linearity(
                tf.reduce_sum(tf.multiply(att_coefs, t_nodes)))

        return out

    @tf.function
    def attention_coef(self, node):

        # store vecotrs after the W transformations
        t_nodes = tf.Variable(
            (len(self.neighbors[node])+1, len(self.features[node])), dtype=tf.float32)
        conc_nodes = tf.Variable(
            (len(self.neighbors[node])+1, 2*len(self.features[node])), dtype=tf.float32)
        # array that stores the att coefs for the node with all its neighbors
        att_coef = tf.Variable(
            (len(self.neighbors[node])+1, 1), dtype=tf.float32)

        # W trasformations (with also the node itself)
        t_nodes[-1] = tf.matmul(self.W, self.features[node])
        conc_nodes[-1] = tf.concat((t_nodes[-1], t_nodes[-1]), axis=0)
        for idx, j in enumerate(self.neighbors[node]):
            t_nodes[idx] = tf.matmul(self.W, self.features[j])
            conc_nodes[idx] = tf.concat((t_nodes[-1], t_nodes[idx]), axis=0)

        den = 1e-7  # for stability
        for k in conc_nodes:
            den += tf.exp(self.lrelu(tf.matmul(tf.transpose(self.a), k)))

        for idx in range(len(self.neighbors[node])+1):
            att_coef[idx] = tf.exp(self.lrelu(
                tf.matmul(self.a, conc_nodes[idx]))) / den

        del conc_nodes

        return att_coef, t_nodes

 """
