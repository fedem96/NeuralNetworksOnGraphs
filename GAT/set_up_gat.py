import tensorflow as tf
import numpy as np
import sys
import os
import datetime
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from GAT.train import train
from GAT.models import GAT
from GAT.metrics import RegularizedLoss, EarlyStop

from utils import read_dataset, permute, split, one_hot_enc, adjacency_matrix


def set_up_gat(dataset, epochs, val_period, log):

    print("GAT!")

    if dataset == 'pubmed':
        lr = 1e-2
        nheads = [8,8]
        l2_weight = 1e-3
    else:
        lr = 5e-3
        nheads = [8,1]
        l2_weight = 5e-4    

    drop_rate = 0.6
    patience = 2
    nhidden = 8

    # Preprocess on data
    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(
        features, neighbors, labels, o_h_labels, keys)
    print("obtaining masks")
    train_idx, val_idx, test_idx = split(dataset, labels)
    
    print("adjacency matrix")
    # sparse csr matrix
    graph = adjacency_matrix(neighbors, self_loops=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = RegularizedLoss(l2_weight)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

    model = GAT(graph, len(o_h_labels[0]), nhidden, nheads, drop_rate)

    sched_acc = EarlyStop(model, monitor=val_acc, patience=patience, save_model=False)
    sched_loss = EarlyStop(model, monitor=val_loss, patience=patience, save_model=False)

    print("begin training")

    train(model, features, o_h_labels, train_idx, val_idx, epochs, optimizer, loss_fn,
          train_loss, train_accuracy, val_loss, val_acc, [sched_acc, sched_loss], val_period)

    # test()


if __name__ == '__main__':

    dataset = "cora"        # "cora" "pubmed" "citeseer"

    epochs = 50
    val_period = 1          # each epoch validation
    log = 1                 # every two epochs print train loss and acc

    data_seed = 0
    net_seed = 0
    tf.random.set_seed(net_seed)
    np.random.seed(data_seed)

    set_up_gat(dataset, epochs, val_period, log)
