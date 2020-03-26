import tensorflow as tf
import numpy as np
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import read_dataset, permute, split, one_hot_enc
from Planetoid.planetoid_t import Planetoid_T
from Planetoid.planetoid_i import Planetoid_I

class UnlabeldLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_true, axis=1)
        dot_prod = tf.math.multiply(s, y_pred)
        # -tf.nn.softplus(-x)  # FIXME: https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        return -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))


def main():

    # r1, r2, q, d, N1, N2, T1, T2 #FIXME: trova veri valori
    args = [0.7, 5/6, 10, 3, 200, 200, 20, 20]
    embedding_size = 50
    dataset = "pubmed"
    seed = 1234
    modality = "I"  # can be T (transductive) or I (inductive)

    print("Planetoid-{:s}!".format(modality))

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    permute(features, neighbors, labels, o_h_labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset, len(features))

    # Define model, losses and optimizers
    if modality == "I":
        model = Planetoid_I(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)
    elif modality == "T":
        model = Planetoid_T(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)

    L_s = tf.keras.losses.CategoricalCrossentropy()
    L_u = UnlabeldLoss(args[-1])
    optimizer_u = tf.keras.optimizers.SGD(learning_rate=1e-2)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=1e-1)

    # png_name = "Planetoid-"+modality+".png"
    # tf.keras.utils.plot_model(model, png_name, show_shapes=True)

    # Pretrain iterations
    model.pretrain_step(L_u, optimizer_u, 20)

    # Train model
    model.step_train(L_s, L_u, optimizer_u,
                     optimizer_s, args[-2], args[-1])


if __name__ == '__main__':

    main()
