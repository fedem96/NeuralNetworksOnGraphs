import tensorflow as tf
import numpy as np
import datetime, os
from tqdm import tqdm

from collections import defaultdict as dd

from time import time 

class Planetoid(tf.keras.Model):

    def __init__(self, A, labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate, mask_train, labeled_iters):
        super().__init__()
        self.A = A
        self.labels = labels
        self.features_size = len(labels)
        self.labels_size = len(labels[0])
        self.labeled_iters = labeled_iters
        self.embedding_size = embedding_dim
        self.q = random_walk_length
        self.d = window_size
        self.r1 = neg_sample
        self.r2 = sample_context_rate
        self.mask_train = mask_train
        self.train_size = len(np.where(mask_train)[0])  # FIXME: SISTEMA NON QUA 

    def call(self):
        return

    # def sample_context(self, node, perm, perm_train, labels, label2inst, not_label, it=12000):
    #     """ Algorithm 1: Sample graph context for one node """
    #     max_index = max(perm)
    #     g = []
    #     g1 = []
    #     gy = []

    #     # if it >= self.labeled_iters:
    #     #     random_walk = [node]
    #     #     for _ in range(self.q):
    #     #         random_walk.append(np.random.choice(self.A[random_walk[-1]].indices))
    #     #     for l in range(len(random_walk)):
    #     #         if random_walk[l] >= max_index: continue
    #     #         for m in range(l - self.d, l + self.d + 1):
    #     #             if m < 0 or m >= len(random_walk): continue
    #     #             if random_walk[m] >= max_index: continue
    #     #             g.append([random_walk[l]])
    #     #             g1.append([random_walk[m]])
    #     #             gy.append([1.0])
    #     #             for _ in range(self.r1):
    #     #                 g.append([random_walk[l]])
    #     #                 g1.append([np.random.choice(perm)])
    #     #                 gy.append([- 1.0])

    #     # else:
    #     #     i = np.random.choice(perm_train)
    #     #     c = np.random.choice(label2inst[labels[i]])
    #     #     g.append([i])
    #     #     g1.append([c])
    #     #     gy.append([1.0])
    #     #     for _ in range(self.r1):
    #     #         g.append([i])
    #     #         c = np.random.choice(not_label[labels[i]])
    #     #         g1.append([c])
    #     #         gy.append([- 1.0])

    #     # g = np.array(g, dtype = np.int32)
    #     # g1 = np.array(g1, dtype = np.int32)
    #     # gy = np.array(gy, dtype = np.float32)
    #     batch_inst = []
    #     batch_labels = []
    #     if it >= max_iters:
    #         random_walk = [n]
    #         for _ in range(1,rl):
    #             random_walk.append(np.random.choice(A[random_walk[-1]].indices))
    #         for l in range(len(random_walk)):
    #             i = random_walk[l]
    #             if i >= max_index: continue
    #             for m in range(max(0,l - ws), min(l + ws + 1, rl)):
    #                 if random_walk[m] >= max_index: continue
    #                 batch_inst.append([i,random_walk[m]])
    #                 batch_labels.append(1.0)
    #                 for _ in range(ns):
    #                     batch_inst.append([i, np.random.choice(perm)])
    #                     batch_labels.append(- 1.0)

    #     else:
    #         i = np.random.choice(perm_train)
    #         batch_inst.append([i, np.random.choice(label2inst[labels[i]])])
    #         batch_labels.append(1.0)
    #         for _ in range(ns):
    #             batch_inst.append([i, np.random.choice(not_label[labels[i]])])
    #             batch_labels.append(- 1.0)
    #     return g, g1, gy

    def labeled_batch(self, features, labels, mask_train, N1):
        """ Generate mini-batch for labeled nodes """
        while True:
            perm = np.array(np.random.permutation(np.where(self.mask_train)[0]), dtype=np.int32)
            j = 0
            while j < self.train_size:
                k = min(self.train_size, j+N1)
                b_x = features[perm[j:k]]
                b_y = self.labels[perm[j:k]]
                yield b_x, b_y, perm[j:k]
                j = k

    def compute_iters(self, it):
        if it < 1:
            it = 1 if np.random.random() < it else 0
        return int(it) 

    def train(self, features, labels, mask_train, mask_val, mask_test, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, 
        val_accuracy, train_loss, train_loss_u, val_loss, T1, T2, N1, N2, patience, checkpoint_path=None, verbose=1):
        
        # logs for train, val accuracy and loss
        file_writer = tf.summary.create_file_writer("./logs/train/")
        file_writer.set_as_default()

        max_t_acc = 0
        best_weights = None
        if not checkpoint_path==None:
            ckpt_name = 'Planetoid_ckpts/cp.ckpt'
            checkpoint_path = os.path.join(checkpoint_path, ckpt_name)

        # for epoch in tqdm(range(1, epochs+1)):
        for epoch in range(1, epochs+1):

            if verbose > 0: print("Epoch: {:d} ".format(epoch), end=' ')

            loss_s = self.train_step(features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2)

            if verbose > 0: print("loss {:.3f}, acc {:.3f} ==>".format(loss_s, train_accuracy.result()), end=' ')

            loss = self.eval(features, labels, mask_val, L_s, val_accuracy, val_loss)

            if verbose > 0: print(" val loss {:.3f} val acc: {:.3f}".format(loss, val_accuracy.result()))
        
            if val_accuracy.result() > max_t_acc:
                max_t_acc = val_accuracy.result()
                bw_epoch = epoch
                best_weights = self.get_weights()

                # write scalars only when acc increases
                tf.summary.scalar('bw_loss', data=train_loss.result(), step=epoch)
                tf.summary.scalar('bw_accuracy', data=train_accuracy.result(), step=epoch)
                tf.summary.scalar('bw_epoch', data=epoch, step=epoch)
                tf.summary.scalar('bw_val_loss', data=val_loss.result(), step=epoch)
                tf.summary.scalar('bw_val_accuracy', data=val_accuracy.result(), step=epoch)

                if not checkpoint_path==None: self.save_weights(checkpoint_path)
                ep_wait = 0
            elif patience > 0:      # patience < 0 means no early stopping
                ep_wait += 1
                if ep_wait >= patience: 
                    if verbose > 0: print("Early stop at epoch {:d}, best val acc {:03f}".format(epoch, max_t_acc))
                    break
            

            # write scalars only when acc increases
            tf.summary.scalar('loss', data=train_loss.result(), step=epoch)
            tf.summary.scalar('masked_accuracy', data=train_accuracy.result(), step=epoch)
            tf.summary.scalar('val_loss', data=val_loss.result(), step=epoch)
            tf.summary.scalar('val_masked_accuracy', data=val_accuracy.result(), step=epoch)


            # Reset metrics every epoch
            # train_loss.reset_states() # FIXME: NON LA USO
            # train_loss_u.reset_states() # FIXME: NON LA USO
            # val_loss.reset_states() # FIXME: NON LA USO
            train_accuracy.reset_states()
            val_accuracy.reset_states()

        self.set_weights(best_weights)

        print("Best val acc {:03f} at epoch {:d}".format(max_t_acc, bw_epoch))
        

    def test(self, features, labels, mask_test, L_s, test_accuracy, test_loss):

        file_writer = tf.summary.create_file_writer("./logs/test/")
        file_writer.set_as_default()

        self.eval(features, labels, mask_test, L_s, test_accuracy, test_loss)

        # write scalars
        tf.summary.scalar('bw_test_loss', data=test_loss.result(), step=1)
        tf.summary.scalar('bw_test_accuracy', data=test_accuracy.result(), step=1)

        return test_loss.result(), test_accuracy.result()


class Planetoid_T(Planetoid):
    """ Planetoid transductive """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(
            self.labels_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Embedding layers for graph context
        self.emb_inst = tf.keras.layers.Embedding(self.features_size, self.embedding_size)

        if self.r1 > 0:
            # negative sample embedding
            self.emb_cont = tf.keras.layers.Embedding(self.features_size, self.embedding_size)
        else:
            # graph context with all nodes
            self.emb_cont = tf.keras.layers.Dense(self.features_size, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

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
            # freeze embedding during label classification
            self.emb_cont.trainable = False
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = True

            h_f = self.h_k(inputs[0])

            embs = self.emb_inst(inputs[1])
            h_e = self.h_l(embs)

            h_node = tf.keras.layers.concatenate([h_f, h_e])
            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            # enable only embedding layer during unsupervised learning
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = False
            self.emb_cont.trainable = True

            emb_i = self.emb_inst(inputs[:, 0])
            
            if self.r1 > 0:
                # negative sample enabled
                emb_c = self.emb_cont(inputs[:, 1])
                out = tf.multiply(emb_i, emb_c)
            else: 
                # softmax layer
                out = self.emb_cont(emb_i)

            return out

    def context_batch(self, N2, it):
        """ Algorithm 1: Sampling graph context (with negative sample) """
        # while True:
        #     perm = np.random.permutation(self.features_size)
        #     perm_train = np.random.permutation(np.where(self.mask_train)[0])
        #     j = 0
        #     while j < len(perm):
        #         N2 = N2//2 if it<2000 else N2
        #         k = min(len(perm), j+N2)
        #         for idx,n in enumerate(perm[j:k]):
        #             i, c, gamma = self.sample_context(n, perm, perm_train, it)
        #             if idx == 0:
        #                 context_b_x = np.concatenate((i,c),-1)
        #                 context_b_y = gamma
        #             else:
        #                 context_b_x = np.vstack((context_b_x, np.concatenate((i,c),-1)))
        #                 context_b_y = np.vstack((context_b_y,gamma))

        #         yield np.array(context_b_x, dtype=np.int32), np.array(context_b_y, dtype=np.float32)
        #         j = k
        max_iters = self.labeled_iters
        A = self.A
        rl = self.q
        ws = self.d
        ns = self.r1
        choice = np.random.choice
        train_idx = np.where(self.mask_train)[0]

        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in train_idx:
            flag = False
            for j in range(self.labels_size):
                if self.labels[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.labels[i, j] == 0:
                    not_label[j].append(i)

        while True:
            perm = np.random.permutation(self.features_size)
            perm_train = np.random.permutation(train_idx)
            max_index = max(perm)
            j = 0
            while j < len(perm):
                N2 = N2//2 if it<2000 else N2
                k = min(len(perm), j+N2)
                batch_inst = []
                batch_labels = []
                for n in perm[j:k]:
                    if it >= max_iters:
                        random_walk = [n]
                        for _ in range(1,rl):
                            random_walk.append(choice(A[random_walk[-1]].indices))
                        for l in range(len(random_walk)):
                            i = random_walk[l]
                            if i >= max_index: continue
                            for m in range(max(0,l - ws), min(l + ws + 1, rl)):
                                if random_walk[m] >= max_index: continue
                                batch_inst.append([i,random_walk[m]])
                                batch_labels.append(1.0)
                                for _ in range(ns):
                                    batch_inst.append([i, choice(perm)])
                                    batch_labels.append(- 1.0)

                    else:
                        i = choice(perm_train)
                        batch_inst.append([i, choice(label2inst[labels[i]])])
                        batch_labels.append(1.0)
                        for _ in range(ns):
                            batch_inst.append([i, choice(not_label[labels[i]])])
                            batch_labels.append(- 1.0)
                
                yield np.array(batch_inst, dtype=np.int32), np.array(batch_labels, dtype=np.float32)
                j = k

    
    def train_step(self, features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2):
        """ One train epoch: graph context and label classification """
        loss_s = 0
        loss_u = 0
        
        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, indices = next(self.labeled_batch(features, labels, mask_train, N1))
            with tf.GradientTape() as tape:
                out = self.call([b_x, indices], modality="s")
                # loss_s = L_s(b_y, out)
                loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(b_y, out))
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))

            # train_loss(loss_s)
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_y = next(self.context_batch(N2, it))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                if self.r1>0:
                    loss_u = L_u(b_y, out)
                else:
                    loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(b_x[:,1], out))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

        return loss_s
            # train_loss_u(loss_u)

    def pretrain_step(self, features, mask_test, L_u, optimizer_u, train_loss_u, iters, N2):

        # for it in tqdm(range(1, iters+1)):
        for it in range(1, iters+1):
            b_x, b_y = next(self.context_batch(N2, it))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="u")
                if self.r1>0:
                    loss_u = L_u(b_y, out)
                else:
                    loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(b_x[:,1], out))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            print(it, loss_u.numpy())
            # train_loss_u(loss_u)

    def eval(self, features, labels, mask, L_s, test_accuracy, test_loss):
                
        indices_test = np.where(mask==True)[0]
        predictions = self.call([features[mask], indices_test], modality="s")
        loss = L_s(labels[mask], predictions)

        # test_loss(loss)
        test_accuracy(labels[mask], predictions)

        return loss


class Planetoid_I(Planetoid):
    """ Planetoid inductive """

    def __init__(self, mask_test, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.size_valid_ind = len(np.where(mask_test==False)[0])

        # Hidden features representations
        self.h_k = tf.keras.layers.Dense(self.labels_size, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Parametric Embedding for graph context
        self.par_embedding = tf.keras.layers.Dense(self.embedding_size, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform)
        
        if self.r1 > 0:
            # negative sample embedding
            self.emb_cont = tf.keras.layers.Embedding(self.size_valid_ind, self.embedding_size)
        else:
            # graph context with all nodes
            self.emb_cont = tf.keras.layers.Dense(self.size_valid_ind, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

        # Hidden embedding representations
        self.h_l = tf.keras.layers.Dense(self.labels_size, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform)

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
            self.par_embedding.trainable = False
            self.emb_cont.trainable = False
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable = True 

            h_f = self.h_k(inputs)

            h_l1 = self.par_embedding(inputs)
            h_e = self.h_l(h_l1)

            h_node = tf.keras.layers.concatenate([h_f, h_e])

            out = self.pred_layer(h_node)

            return out

        elif modality == "u":
            # freeze some layers 
            self.h_k.trainable = self.h_l.trainable = self.pred_layer.trainable =  False
            self.emb_cont.trainable = True
            self.par_embedding.trainable = True

            emb_i = self.par_embedding(inputs[0])

            if self.r1 > 0:
                # negative sample enabled
                emb_c = self.emb_cont(inputs[1])
                out = tf.multiply(emb_i, emb_c)
            else:
                # softmax layer
                out = self.emb_cont(emb_i)

            return out

    # def context_batch(self, features, mask_test, N2, it):
    #     """ Algorithm 1: Sampling graph context (with negative sample) """
    #     train_idx = np.where(self.mask_train)[0]
    #     labels, label2inst, not_label = [], dd(list), dd(list)
    #     for i in train_idx:
    #         flag = False
    #         for j in range(self.labels_size):
    #             if self.labels[i, j] == 1 and not flag:
    #                 labels.append(j)
    #                 label2inst[j].append(i)
    #                 flag = True
    #             elif self.labels[i, j] == 0:
    #                 not_label[j].append(i)

    #     while True:
    #         perm = np.random.permutation(self.size_valid_ind)
    #         perm_train = np.random.permutation(train_idx)
    #         j = 0
    #         while j < len(perm):
    #             context_b_x, context_b_y = [], []
    #             k = min(len(perm), j+N2)
    #             for idx,n in enumerate(perm[j:k]):
    #                 i, c, gamma = self.sample_context(n, perm, perm_train, labels, label2inst, not_label, it)
    #                 if idx == 0:
    #                     context_b_x = np.concatenate((i,c),-1)
    #                     context_b_y = gamma
    #                 else:
    #                     context_b_x = np.vstack((context_b_x, np.concatenate((i,c),-1)))
    #                     context_b_y = np.vstack((context_b_y,gamma))

    #             context_b_x = np.array(context_b_x, dtype=np.int32)
    #             yield features[context_b_x[:,0]], context_b_x[:,1], np.array(context_b_y, dtype=np.float32)
    #             j = k

    def context_batch(self, features, mask_test, N2, it=12000):
        """ Algorithm 1: Sampling graph context (with negative sample) """

        max_iters = self.labeled_iters
        A = self.A
        rl = self.q
        ws = self.d
        ns = self.r1
        choice = np.random.choice

        train_idx = np.where(self.mask_train)[0]
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in train_idx:
            flag = False
            for j in range(self.labels_size):
                if self.labels[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.labels[i, j] == 0:
                    not_label[j].append(i)

        while True:
            perm = np.random.permutation(self.size_valid_ind)
            perm_train = np.random.permutation(train_idx)
            max_index = max(perm)
            j = 0
            while j < len(perm):
                k = min(len(perm), j+N2)
                batch_inst = []
                batch_labels = []
                for n in perm[j:k]:
                    if it >= max_iters:
                        random_walk = [n]
                        for _ in range(1,rl):
                            random_walk.append(choice(A[random_walk[-1]].indices))
                        for l in range(len(random_walk)):
                            i = random_walk[l]
                            if i >= max_index: continue
                            for m in range(max(0,l - ws), min(l + ws + 1, rl)):
                                if random_walk[m] >= max_index: continue
                                batch_inst.append([i,random_walk[m]])
                                batch_labels.append(1.0)
                                for _ in range(ns):
                                    batch_inst.append([i, choice(perm)])
                                    batch_labels.append(- 1.0)

                    else:
                        i = choice(perm_train)
                        batch_inst.append([i, choice(label2inst[labels[i]])])
                        batch_labels.append(1.0)
                        for _ in range(ns):
                            batch_inst.append([i, choice(not_label[labels[i]])])
                            batch_labels.append(- 1.0)

                batch_inst = np.array(batch_inst, dtype=np.int32)
                yield features[batch_inst[:,0]], batch_inst[:,1], batch_labels
                j = k


    
    def train_step(self, features, labels, mask_train, mask_test, L_s, L_u, optimizer_u, optimizer_s,
                train_accuracy, train_loss, train_loss_u, T1, T2, N1, N2):
        """ One train epoch: graph context and label classification """
        loss_s = 0
        loss_u = 0

        for it in range(1, self.compute_iters(T1)+1):
            b_x, b_y, _ = next(self.labeled_batch(features, labels, mask_train, N1))
            with tf.GradientTape() as tape:
                out = self.call(b_x, modality="s")
                # loss_s = L_s(b_y,out)
                loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(b_y, out))
            grads = tape.gradient(loss_s, self.trainable_weights)
            optimizer_s.apply_gradients(zip(grads, self.trainable_weights))
        
            # train_loss(loss_s)    
            train_accuracy(b_y, out)

        for it in range(1, self.compute_iters(T2)+1):
            b_x, b_c, b_y = next(self.context_batch(features, mask_test, N2, it))
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                if self.r1>0:
                    loss_u = L_u(b_y, out)
                else:
                    loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(b_c, out))
                # target = b_y if self.r1>0 else b_c
                # loss_u = L_u(target, out)
                # loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(target, out))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))

        return loss_s
            # train_loss_u(loss_u)

    def pretrain_step(self, features, mask_test, L_u, optimizer_u, train_loss_u, iters, N2):
        
        # for it in tqdm(range(1, iters+1)):
        for it in range(1, iters+1):
            b_x, b_c, b_y = next(self.context_batch(features, mask_test, N2, it))
            with tf.GradientTape() as tape:
                out = self.call([b_x, b_c], modality="u")
                if self.r1>0:
                    loss_u = L_u(b_y, out)
                else:
                    loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(b_c, out))
                # target = b_y if self.r1>0 else b_c
                # # loss_u = L_u(target, out)
                # loss_u = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(target, out))
            grads = tape.gradient(loss_u, self.trainable_weights)
            optimizer_u.apply_gradients(zip(grads, self.trainable_weights))
            print(it, loss_u.numpy())
            # train_loss_u(loss_u)

    def eval(self, features, labels, mask, L_s, test_accuracy, test_loss):
    
        predictions = self.call(features[mask], modality="s")
        loss = L_s(labels[mask], predictions)

        test_accuracy(labels[mask], predictions)    
        return loss