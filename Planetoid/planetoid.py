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

        self.r1 = args[0]#args.r1
        self.r2 = args[1]#args.r2
        self.q  = args[2]#args.q
        self.d = args[3]#args.d
        self.N1 = args[4]#args.N1
        self.N2 = args[5]#args.N2
        self.T1 = args[6]#args.T1
        self.T2 = args[7]#args.T2


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
                           if len(self.neighbors[indices[el[0]]]) > 0]
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
                while self.labels[indices[i]] != self.labels[indices[c]]:
                    c = np.random.choice(indices)
            elif gamma == -1:
                i, c = np.random.choice(indices, 2)
                while (self.labels[indices[i]] == self.labels[indices[c]]).all():
                    c = np.random.choice(indices)

        return i, c, gamma



