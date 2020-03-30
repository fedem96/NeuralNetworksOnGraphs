import tensorflow as tf
import numpy as np
import sys
import os
import datetime
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from utils import read_dataset, permute, split, one_hot_enc
from GAT.train import train, test
from GAT.layers import GAT

def set_up_gat(dataset, epochs, batch_size, val_period, log, seed):

    print("GAT!")

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset, labels)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

    if dataset == 'pubmed':
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    else: 
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
    
    sched_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    sched_acc = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=100)
    # tensorboard = tf.keras.callbacks.Tensorboard(log_dir='.logs/')

    if dataset == 'pubmed':
        n_output_heads = 8
    else:
        n_output_heads = 1

    model = GAT(neighbors, len(o_h_labels[0]), n_output_heads=n_output_heads)

    train(model, features, o_h_labels, features[val_idx], o_h_labels[val_idx],
        epochs, optimizer, loss_fn, train_loss, train_accuracy, val_loss, val_acc, val_period)
    
    test()


if __name__ == '__main__':

    dataset = "cora"    # "cora" "pubmed" "citeseer"

    epochs = 10
    val_period = 50        # each epoch validation
    log = 1                 # every two epochs print train loss and acc
    
    batch_size = 32
    seed = 1234   

    set_up_gat(dataset, epochs, batch_size, val_period, log, seed)