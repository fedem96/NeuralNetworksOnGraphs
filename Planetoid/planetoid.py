import tensorflow as tf
import numpy as np


class Planetoid(tf.keras.Model):

    def __init__(self, features, neighbors, labels, embedding_size, mask_train, mask_val, mask_test, args):
        super().__init__()
        self.features = features
        self.neighbors = neighbors
        self.labels = labels
        self.labels_size = len(labels[0])      # FIXME: dovrebbe essere lo spazio delle classi 3 / 6 / 7 a seconda dataset pensandole in one hot encode
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



