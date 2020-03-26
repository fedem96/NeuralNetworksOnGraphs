import tensorflow as tf
import numpy as np

from planetoid_t import Planetoid_T
from ut_pl import read_dataset, permute, split, one_hot_enc


class UnlabeldLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_true, axis=1)
        dot_prod = tf.math.multiply(s, y_pred)
        # -tf.nn.softplus(-x)  # FIXME: https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        return -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))


def train(model, loss_s, loss_u, optimizer_u, optimizer_s, T1, T2):
    """ One train iteration: graph context and label classification """
    loss = 0
    for epoch in range(1, T1+1):
        b_x, b_y, indices = next(model.labeled_batch())
        with tf.GradientTape() as tape:
            out = model([b_x, indices], modality="s")
            loss += loss_s(out, b_y)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer_s.apply_gradients(zip(grads, model.trainable_weights))
    
    loss = 0
    for epoch in range(1, T2+1):
        b_x, b_y = next(model.context_batch())
        with tf.GradientTape() as tape:
            out = model(b_x, modality="u")
            loss += loss_u(out, b_y)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer_u.apply_gradients(zip(grads, model.trainable_weights))

    return


if __name__ == '__main__':

    # r1, r2, q, d, N1, N2, T1, T2 #FIXME: trova veri valori
    args = [0.7, 5/6, 10, 3, 200, 200, 20, 20]
    embedding_size = 50
    dataset = "pubmed"
    seed = 1234

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    permute(features, neighbors, labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset)

    # Define model loss and optimizer
    planetoid_t = Planetoid_T(
        features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)
    loss_s = tf.keras.losses.CategoricalCrossentropy()
    loss_u = UnlabeldLoss(args[-1])
    optimizer_u = tf.keras.optimizers.SGD(learning_rate=1e-1)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=1e-2)

    # Train model
    train(planetoid_t, loss_s, loss_u, optimizer_u, optimizer_s, args[-2], args[-1])
