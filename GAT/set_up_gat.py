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

class RegularizedLoss(tf.keras.losses.Loss):

    def __init__(self, l2_weight):
        super().__init__()
        self.l2_weight = l2_weight

    def call(self, y_true, y_pred):
        loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        l2_loss = tf.nn.l2_loss(y_pred-y_true)
        return loss + self.l2_weight * l2_loss


class EarlyStop(tf.keras.callbacks.Callback):

    def __init__(self, model, monitor, patience=0, save_model=False, checkpoint_path='./'):
        super(EarlyStop, self).__init__()
        self.model = model
        self.monitor = monitor
        self.modality = 'acc' if 'acc' in monitor.name else 'loss'
        self.patience = patience
        self.best_weights = None
        self.save_model = save_model
        if self.save_model:
            ckpt_name = self.model.name + 'ckpts/cp-{epoch:03d}-' + self.modality + '-{v:03f}.ckpt'
            self.checkpoint_path = os.path.join(checkpoint_path, ckpt_name)

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf if self.modality == 'loss' else 0

    def on_epoch_end(self, epoch):
        current = self.monitor.result()
        if (np.less(current, self.best) and self.modality == 'acc') or (np.greater(current, self.best) and self.modality == 'loss'):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            if self.save_model:
                self.model.save_weights(self.checkpoint_path.format(
                    epoch=epoch, v=self.best))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def set_up_gat(dataset, epochs, l2_weight, val_period, log, seed):

    print("GAT!")

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    features, neighbors, labels, o_h_labels, keys = permute(
        features, neighbors, labels, o_h_labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset, labels)

    loss_fn = RegularizedLoss(l2_weight)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

    if dataset == 'pubmed':
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        
    if dataset == 'pubmed':
        n_output_heads = 8
    else:
        n_output_heads = 1

    model = GAT(neighbors, len(o_h_labels[0]), n_output_heads=n_output_heads)

    sched_acc = EarlyStop(model, monitor=val_acc, patience=100)
    sched_loss = EarlyStop(model, monitor=val_loss, patience=100)

    train(model, features, o_h_labels, train_idx, val_idx, epochs, optimizer, loss_fn,
          train_loss, train_accuracy, val_loss, val_acc, [sched_acc, sched_loss], val_period)

    test()


if __name__ == '__main__':

    dataset = "cora"    # "cora" "pubmed" "citeseer"

    epochs = 10
    val_period = 1        # each epoch validation
    log = 1                 # every two epochs print train loss and acc

    l2_weight = 5e-4
    seed = 1234

    set_up_gat(dataset, epochs, l2_weight, val_period, log, seed)
