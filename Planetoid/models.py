import tensorflow as tf
import numpy as np
import datetime

class Planetoid(tf.keras.Model):

    def __init__(self, neighbors, labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate):
        super().__init__()
        self.neighbors = neighbors
        self.features_size = len(neighbors)
        self.labels = labels
        self.labels_size = len(labels[0])
        self.embedding_size = embedding_dim
        self.q  = random_walk_length
        self.d = window_size
        self.r1 = neg_sample
        self.r2 = sample_context_rate

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

    def labeled_batch(self, features, labels, mask_train, N1):
        """ Generate mini-batch for labeled nodes """
        while True:
            indices = np.random.permutation(len(features[mask_train]))
            j = 0
            while j < len(mask_train):
                k = min(len(mask_train), j+N1)
                b_x = features[indices[j:k]]
                b_y = labels[indices[j:k]]
                yield np.array(b_x, dtype=np.float32), np.array(b_y, dtype=np.float32), indices[j:k]
                j = k

    def compute_iters(self, it):
        if it < 1:
            it = 1 if np.random.random() < it else 0
        return int(it) 

    def train(self, features, labels, mask_train, mask_test, epochs, L_s, L_u, optimizer_u, optimizer_s, 
            train_accuracy, test_accuracy, train_loss, train_loss_u, test_loss, T1, T2, N1, N2):

        max_t_acc = 0
        patience = 20
    
        for epoch in range(1, epochs+1):

            print("Epoch: {:d} ==> ".format(epoch), end=' ')

            self.train_step(features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2)

            print("Train Loss: s {:.3f} u {:.3f}, Train Accuracy: {:.3f}".format(train_loss.result(), train_loss_u.result(), train_accuracy.result()))

            self.eval(features, labels, mask_test, L_s, test_accuracy, test_loss, mode="val")

            print("\nEpoch {:d}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}\n".format(epoch, test_loss.result(), test_accuracy.result()))

            if test_accuracy.result() > max_t_acc:
                max_t_acc = test_accuracy.result()
                ep_wait = 0
            else:
                ep_wait += 1
                if ep_wait <= patience:
                    break

            # Reset metrics every epoch
            train_loss.reset_states()
            train_loss_u.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()


    def test(self, L_s, test_accuracy, test_loss):

        self.eval(L_s, test_accuracy, test_loss, mode="test")

        print("Test Loss: {:.3f}, Test Accuracy: {:.2f}%".format(test_loss.result(), test_accuracy.result()*100))

        return


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

    def context_batch(self, N2):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        while True:
            indices = np.random.permutation(len(self.neighbors))
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                k = min(len(indices), j+N2)
                for n in indices[j:k]:
                    if len(self.neighbors[n]) == 0:
                        continue    # aka node without neighbors
                    i, c, gamma = self.sample_context(n, indices)
                    context_b_x.append([i, c])
                    context_b_y.append(gamma)

                yield np.array(context_b_x, dtype=np.int32), np.array(context_b_y, dtype=np.int32)
                j = k
    
    def train_step(self, features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2):
        """ One train epoch: graph context and label classification """
        
        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, indices = next(self.labeled_batch(features, labels, mask_train, N1))
            with tf.GradientTape() as tape:
                out = self.call([b_x, indices], modality="s")
                loss_s = L_s(b_y, out)
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))

            train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_y = next(self.context_batch(N2))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

            train_loss_u(loss_u)

    def pretrain_step(self, features, mask_test, L_u, optimizer_u, train_loss_u, iters, N2):

        for it in range(1, iters+1):
            b_x, b_y = next(self.context_batch(N2))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            train_loss_u(loss_u)

    def eval(self, features, labels, mask, L_s, test_accuracy, test_loss, mode="val"):
                
        indices_test = np.where(mask==True)[0]
        predictions = self.call([features[mask], indices_test], modality="s")
        loss = L_s(labels[mask], predictions)

        test_loss(loss)
        test_accuracy(labels[mask], predictions)


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

    def context_batch(self, features, mask_test, N2):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        while True:
            size_valid_ind = len(np.where(mask_test==False)[0])
            indices = np.random.permutation(size_valid_ind)
            j = 0
            while j < len(indices):
                context_b_x, context_b_y = [], []
                l = 0   # at least 5 samples in batch
                while l < 5:
                    k = min(len(indices), j+N2)
                    for n in indices[j:k]:
                        if mask_test[n]:
                            continue    # aka test node
                        if len(self.neighbors[indices[n]]) == 0:
                            continue    # aka node without neighbors
                        i, c, gamma = self.sample_context(n, indices)
                        context_b_x.append([i, c])
                        context_b_y.append(gamma)
                        l+=1

                context_b_x = np.array(context_b_x, dtype=np.int32)
                yield features[context_b_x[:,0]], context_b_x[:,1], np.array(context_b_y, dtype=np.float32)
                j = k

    def train_step(self, features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2):
        """ One train epoch: graph context and label classification """

        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, _ = next(self.labeled_batch(features, labels, mask_train, N1))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="s")
                loss_s = L_s(b_y, out)
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))
        
            train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_c, b_y = next(self.context_batch(features, mask_test, N2))
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

            train_loss_u(loss_u)

    def pretrain_step(self, features, mask_test, L_u, optimizer_u, train_loss_u, iters, N2):
    
        for it in range(1, iters+1):
            b_x, b_c, b_y = next(self.context_batch(features, mask_test, N2))
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                loss_u = L_u(b_y, out)
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            train_loss_u(loss_u)

    def eval(self, features, labels, mask, L_s, test_accuracy, test_loss, mode="val"):

        predictions = self.call(features[mask], modality="s")
        loss = L_s(labels[mask], predictions)

        test_loss(loss)
        test_accuracy(labels[mask], predictions)


