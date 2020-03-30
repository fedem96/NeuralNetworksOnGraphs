import tensorflow as tf
import numpy as np
import sys
import os
import datetime
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import read_dataset, permute, split, one_hot_enc
from Planetoid.planetoid_i import Planetoid_I
from Planetoid.planetoid_t import Planetoid_T
from Planetoid.train import train, test


class UnlabeledLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_pred, axis=1)
        dot_prod = tf.math.multiply(s, y_true)
        # Credits to https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        # loss = -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))
        loss = tf.reduce_sum(tf.nn.softplus(-dot_prod))
        return loss


def set_up_planetoid(embedding_size, dataset, seed, modality, epochs, val_period, log, pre_train_iters, args):

    print("Planetoid-{:s}!".format(modality))

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset, labels)

    # Define model, loss, metrics and optimizers
    if modality == "I":
        model = Planetoid_I(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)
    elif modality == "T":
        model = Planetoid_T(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)

    L_s = tf.keras.losses.CategoricalCrossentropy()
    L_u = UnlabeledLoss(args['n2'])

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_acc")
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_u = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    optimizer_u = tf.keras.optimizers.SGD(learning_rate=1e-2)       # , momentum=0.99)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=1e-1)       # , momentum=0.99)


    # Pretrain iterations on graph context
    model.pretrain_step(L_u, optimizer_u, train_loss_u, pre_train_iters)

    # Train model    
    train(model, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, test_accuracy, train_loss, train_loss_u, test_loss,
         args['t1'], args['t2'], val_period, log)

    # Test model 
    test(model, L_s, test_accuracy, test_loss)


if __name__ == '__main__':

    dataset = "cora"    # "cora" "pubmed" "citeseer"
    modality = "I"          # can be T (transductive) or I (inductive)    
    epochs = 100
    val_period = 5        # each epoch validation
    log = 1                 # every two epochs print train loss and acc
    pre_train_iters = 100    # graph context pretrain iterations
    
    embedding_size = 50    
    seed = 1234    

    args = {'r1': 5/6, 'r2': 5/6, 'q':10 , 'd':3, 'n1':200, 'n2':200, 't1':20, 't2':20}

    set_up_planetoid(embedding_size, dataset, seed, modality, epochs, val_period, log, pre_train_iters, args)

    