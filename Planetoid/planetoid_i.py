import tensorflow as tf
import numpy as np

from planetoid import Planetoid


class Planetoid_I(Planetoid):
    ''' Planetoid inductive '''

    def __init__(self):
        super().__init__()

        # FIXME: data si aggiunge?

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Parametric Embedding for graph context
        self.par_embedding = tf.keras.layers.Dense(
            self.embedding_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Hidden embedding representations
        self.h_l = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

    def call(self, inputs, modality="s"):
        """ 
            Model forward. Modality specify: 
                - "s" : supervised
                - "u" : unsupervised
        """
        if modality == "s":
            h_f = self.h_k(inputs[0])

            # FIXME: lui vorrebbe l'indice permutato rispetto a quello iniziale
            h_e = self.embedding(inputs[1])

            h_node = tf.keras.layers.concatenate([h_f, h_e])
            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            emb_in = self.embedding(inputs[:, 0])
            emb_out = self.embedding(inputs[:, 1])

            out = tf.multiply(emb_in, emb_out)
            return out

    def context_batch(self):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        while True:
            indices = np.random.permutation(len(self.features))
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                k = min(len(indices), j+self.N2)
                for n in indices[j:k]:
                    if n in self.mask_test:
                        continue    # aka test node
                    if len(self.neighbors[indices[n]]) == 0:
                        continue    # aka node without neighbors
                    i, c, gamma = self.sample_context(n, indices)
                    context_b_x.append([i, c])
                    context_b_y.append(gamma)

                yield np.array(context_b_x, dtype=np.float32), np.array(context_b_y, dtype=np.float32)
                j = k

    def labeled_batch(self):
        """ Generate mini-batch for labeled nodes """
        while True:
            indices = np.random.permutation(len(self.features[self.mask_train]))
            j = 0
            while j < len(self.mask_train):
                k = min(len(self.mask_train), j+self.N1)
                b_x = self.features[indices[j:k]]
                b_y = self.labels[indices[j:k]]
                yield np.array(b_x, dtype=np.float32), np.array(b_y, dtype=np.float32), indices[j:k]
                j = k
