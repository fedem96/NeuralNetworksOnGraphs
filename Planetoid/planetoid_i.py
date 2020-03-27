import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.planetoid import Planetoid


class Planetoid_I(Planetoid):
    """ Planetoid inductive """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Parametric Embedding for graph context
        self.par_embedding = tf.keras.layers.Dense(self.embedding_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)
        
        # Embedding for graph context
        self.embedding = tf.keras.layers.Embedding(self.features_size, self.embedding_size)

        # Hidden embedding representations
        self.h_l = tf.keras.layers.Dense(self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Output layer after concatenation
        self.pred_layer = tf.keras.layers.Dense(self.labels_size, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

    def call(self, inputs, modality="s"):
        """ 
            Model forward. Modality specify: 
                - "s" : supervised
                - "u" : unsupervised
        """
        if modality == "s":
            # freeze embedding graph context layer
            self.embedding.trainable = False
            self.h_k.trainable, self.h_l.trainable, self.pred_layer.trainable = True, True, True

            h_f = self.h_k(inputs)

            h_l1 = self.par_embedding(inputs)
            h_e = self.h_l(h_l1)

            h_node = tf.keras.layers.concatenate([h_f, h_e])

            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            # freeze some layers 
            self.h_k.trainable, self.h_l.trainable, self.pred_layer.trainable = False, False, False
            self.embedding.trainable = True

            emb_in = self.par_embedding(inputs[0])
            emb_out = self.embedding(inputs[1])

            out = tf.multiply(emb_in, emb_out)
            return out

    def context_batch(self):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        while True:
            size_valid_ind = len(np.where(self.mask_test==False)[0])
            indices = np.random.permutation(size_valid_ind)
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                k = min(len(indices), j+self.N2)
                for n in indices[j:k]:
                    if self.mask_test[n]:
                        continue    # aka test node
                    if len(self.neighbors[indices[n]]) == 0:
                        continue    # aka node without neighbors
                    i, c, gamma = self.sample_context(n, indices)
                    context_b_x.append([i, c])
                    context_b_y.append(gamma)

                context_b_x = np.array(context_b_x, dtype=np.int32)
                yield self.features[context_b_x[:,0]], context_b_x[:,1], np.array(context_b_y, dtype=np.float32)
                j = k

    def train_step(self, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, train_loss, train_loss_u, T1, T2):
        """ One train epoch: graph context and label classification """

        for it in range(1, T1+1):
            b_x, b_y, _ = next(self.labeled_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="s")
                loss_s = L_s(b_y, out)
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))
        
            train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, T2+1):
            b_x, b_c, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

            train_loss_u(loss_u)

    def pretrain_step(self, L_u, optimizer_u, iters):
    
        loss_u = 0
        for it in range(1, iters+1):
            b_x, b_c, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u += L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

        return loss_u

    def test_step(self, L_s, test_accuracy, test_loss, mode="val"):

        mask = self.mask_val if mode == "val" else self.mask_test

        predictions = self.call(self.features[mask], modality="s")
        loss = L_s(self.labels[mask], predictions)

        test_loss(loss)
        test_accuracy(self.labels[mask], predictions)
