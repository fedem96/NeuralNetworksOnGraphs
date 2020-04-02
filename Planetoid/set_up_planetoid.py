import tensorflow as tf
import numpy as np
import sys, os
import datetime

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.models import Planetoid_I, Planetoid_T
from Planetoid.train import train, test
from Planetoid.metrics import UnlabeledLoss
from utils import read_dataset, permute, split, one_hot_enc

def set_up_planetoid(dataset, modality, epochs, val_period, log):

    if modality == "T":
        pre_train_iters = 70    # graph context pretrain iterations
        t1 = 1
        t2 = 0
        lr_u = 1e-2
        n2 = 200
    else: 
        pre_train_iters = 400
        t1 = 1
        t2 = 0.1
        lr_u = 1e-3
        n2 = 20

    n1 = 200
    lr_s = 0.1
    embedding_size = 50
    args = {'r1': 5/6, 'r2': 5/6, 'q':10 , 'd':3, 'n1':n1, 'n2':n2, 't1':t1, 't2':t2}

    print("Planetoid-{:s}!".format(modality))

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
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

    optimizer_u = tf.keras.optimizers.SGD(learning_rate=lr_u)       # , momentum=0.99)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=lr_s)       # , momentum=0.99)

    print("pre-train model")

    # Pretrain iterations on graph context
    model.pretrain_step(L_u, optimizer_u, train_loss_u, pre_train_iters)

    print("train model")
    # Train model    
    train(model, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, 
        test_accuracy, train_loss, train_loss_u, test_loss, args['t1'], args['t2'], val_period, log)

    # Test model 
    test(model, L_s, test_accuracy, test_loss)


if __name__ == '__main__':

    dataset = "pubmed"    # "cora" "pubmed" "citeseer"
    modality = "I"          # can be T (transductive) or I (inductive)    
    epochs = 50
    val_period = 1        # each epoch validation
    log = 1                 # every two epochs print train loss and acc
    
    data_seed = 0
    net_seed = 0
    tf.random.set_seed(net_seed)
    np.random.seed(data_seed)    

    set_up_planetoid(dataset, modality, epochs, val_period, log)

    