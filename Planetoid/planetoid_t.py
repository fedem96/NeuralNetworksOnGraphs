import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.planetoid import Planetoid


class Planetoid_T(Planetoid):
    """ Planetoid transductive """

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

            # freezed during label classification
            embs = self.embedding(inputs[1])
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
            indices = np.random.permutation(len(self.features))
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
    
    def step_train(self, L_s, L_u, optimizer_u, optimizer_s, T1, T2):
        """ One train iteration: graph context and label classification """
        loss_s = 0
        self.embedding.trainable = False
        for epoch in range(1, T1+1):
            b_x, b_y, indices = next(self.labeled_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, indices], modality="s")
                loss_s += tf.reduce_mean(L_s(out, b_y))
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))

        loss_u = 0
        # for l in self.layers:
        #     l.trainable = not l.trainable
        self.h_k.trainable, self.h_l.trainable, self.pred_layer.trainable = False, False, False
        self.embedding.trainable = True
        for epoch in range(1, T2+1):
            b_x, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u += tf.reduce_mean(L_u(out, b_y))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

        return loss_s, loss_u

    def pretrain_step(self, L_u, optimizer_u, iters):

        loss_u = 0
        self.embedding.trainable = True
        for epoch in range(1, iters+1):
            b_x, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u += tf.reduce_mean(L_u(out, b_y))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

        return loss_u
