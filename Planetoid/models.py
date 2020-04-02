import tensorflow as tf
import numpy as np


class Planetoid(tf.keras.Model):

    def __init__(self, features, neighbors, labels, embedding_size, mask_train, mask_val, mask_test, args):
        super().__init__()
        self.features = features
        self.neighbors = neighbors
        self.labels = labels
        self.labels_size = len(labels[0])  
        self.features_size = len(features)
        self.embedding_size = embedding_size

        self.mask_train = mask_train
        self.mask_val = mask_val
        self.mask_test = mask_test

        self.r1 = args['r1']
        self.r2 = args['r2']
        self.q  = args['q']
        self.d = args['d']
        self.N1 = args['n1']
        self.N2 = args['n2']
        self.T1 = args['t1']
        self.T2 = args['t2']


    def call(self):
        return


    def sample_context(self, node, indices):
        """ Algorithm 1: Sample graph context for one node """
        random = np.random.random()

        gamma = 1 if random < self.r1 else -1
        if random < self.r2:
            # random walk from S of length q
            random_walk = [node]
            for _ in range(self.q):
                neigh_size = [el[0] for el in self.neighbors[indices[random_walk[-1]]]
                           if len(self.neighbors[el[0]]) > 0 and el[0] in indices]
                if neigh_size == []:
                    continue
                random_walk.append(np.random.choice(neigh_size))
                # random_walk.append(self.neighbors[np.random.randint(
                #     0, len(self.neighbors[random_walk[-1]]))])

            i = np.random.randint(0, len(random_walk))
            if gamma == 1:
                c = np.random.choice(
                    random_walk[max(0, i-self.d):min(i+self.d+1, len(random_walk))])
            elif gamma == -1:
                c = np.random.choice(indices)
            i = random_walk[i]
        else:
            if gamma == 1:
                i, c = np.random.choice(indices, 2)
                while (self.labels[indices[i]] != self.labels[indices[c]]).any():
                    c = np.random.choice(indices)
            elif gamma == -1:
                i, c = np.random.choice(indices, 2)
                while (self.labels[indices[i]] == self.labels[indices[c]]).all():
                    c = np.random.choice(indices)
            
        return i, c, gamma

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

    def compute_iters(self, it):
        if it < 1:
            it = 1 if np.random.random() < it else 0
        return it 

class Planetoid_T(Planetoid):
    """ Planetoid transductive """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Embedding for graph context
        self.embedding = tf.keras.layers.Embedding(self.features_size, self.embedding_size)

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
            # freeze embedding during label classification
            self.embedding.trainable = False
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = True

            h_f = self.h_k(inputs[0])

            embs = self.embedding(inputs[1])
            h_e = self.h_l(embs)

            h_node = tf.keras.layers.concatenate([h_f, h_e])
            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            # enable only embedding layer during unsupervised learning
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = False
            self.embedding.trainable = True

            emb_i = self.embedding(inputs[:, 0])
            emb_c = self.embedding(inputs[:, 1])

            out = tf.multiply(emb_i, emb_c)
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

                yield np.array(context_b_x, dtype=np.int32), np.array(context_b_y, dtype=np.int32)
                j = k
    
    def train_step(self, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, train_loss, train_loss_u, T1, T2):
        """ One train epoch: graph context and label classification """
        
        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, indices = next(self.labeled_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, indices], modality="s")
                loss_s = L_s(b_y, out)
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))

            train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

            train_loss_u(loss_u)

    def pretrain_step(self, L_u, optimizer_u, train_loss_u, iters):

        for it in range(1, iters+1):
            b_x, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            train_loss_u(loss_u)

    def test_step(self, L_s, test_accuracy, test_loss, mode="val"):

        # mask = self.mask_val if mode == "val" else self.mask_test
        mask = self.mask_test
        
        indices_test = np.where(mask==True)[0]
        predictions = self.call([self.features[mask], indices_test], modality="s")
        loss = L_s(self.labels[mask], predictions)

        test_loss(loss)
        test_accuracy(self.labels[mask], predictions)



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
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = True 

            h_f = self.h_k(inputs)

            h_l1 = self.par_embedding(inputs)
            h_e = self.h_l(h_l1)

            h_node = tf.keras.layers.concatenate([h_f, h_e])

            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            # freeze some layers 
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = False
            self.embedding.trainable = True

            emb_i = self.par_embedding(inputs[0])
            emb_c = self.embedding(inputs[1])

            out = tf.multiply(emb_i, emb_c)

            return out

    def context_batch(self):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        while True:
            size_valid_ind = len(np.where(self.mask_test==False)[0])
            indices = np.random.permutation(size_valid_ind)
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                l = 0   # at least 5 samples in batch
                while l < 5:
                    k = min(len(indices), j+self.N2)
                    for n in indices[j:k]:
                        if self.mask_test[n]:
                            continue    # aka test node
                        if len(self.neighbors[indices[n]]) == 0:
                            continue    # aka node without neighbors
                        i, c, gamma = self.sample_context(n, indices)
                        context_b_x.append([i, c])
                        context_b_y.append(gamma)
                        l+=1

                context_b_x = np.array(context_b_x, dtype=np.int32)
                yield self.features[context_b_x[:,0]], context_b_x[:,1], np.array(context_b_y, dtype=np.float32)
                j = k

    def train_step(self, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, train_loss, train_loss_u, T1, T2):
        """ One train epoch: graph context and label classification """

        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, _ = next(self.labeled_batch())
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="s")
                loss_s = L_s(b_y, out)
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))
        
            train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_c, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

            train_loss_u(loss_u)

    def pretrain_step(self, L_u, optimizer_u, train_loss_u, iters):
    
        for it in range(1, iters+1):
            b_x, b_c, b_y = next(self.context_batch())
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            train_loss_u(loss_u)

    def test_step(self, L_s, test_accuracy, test_loss, mode="val"):

        # mask = self.mask_val if mode == "val" else self.mask_test
        mask = self.mask_test

        predictions = self.call(self.features[mask], modality="s")
        loss = L_s(self.labels[mask], predictions)

        test_loss(loss)
        test_accuracy(self.labels[mask], predictions)


