import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.planetoid import Planetoid


class Planetoid_T(Planetoid):
    ''' Planetoid transductive '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Embedding for graph context
        self.embedding = tf.keras.layers.Embedding(
            self.features_size, self.embedding_size)

        # Hidden embedding representations
        self.h_l = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Output layer after concatenation
        self.pred_layer = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

    def call(self, inputs, modality="s"):
        """ 
            Model forward. Modality specify: 
                - "s" : supervised
                - "u" : unsupervised
        """
        if modality == "s":
            h_f = self.h_k(inputs[0])

            # TODO: METTI TRAINABLE==FALSE
            embs = self.embedding(inputs[1]) # DA TENERE FREEEZATO NEL SUPERVISED
            h_e = self.h_l(embs)

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
            indices = np.random.permutation(len(self.featuresc))
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                k = min(len(indices), j+self.N2)
                for n in indices[j:k]:
                    if len(self.neighbors[n]) == 0:
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
