import tensorflow as tf
import numpy as np
import sys
import os
import datetime
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Planetoid.planetoid_i import Planetoid_I
from Planetoid.planetoid_t import Planetoid_T
from utils import read_dataset, permute, split, one_hot_enc

class UnlabeldLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_pred, axis=1)
        dot_prod = tf.math.multiply(s, y_true)
        # Credits to https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        # loss = -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))
        loss = 1/self.N2 * tf.reduce_sum(tf.nn.softplus(-dot_prod))
        return loss


def train(model, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, test_accuracy, train_loss, train_loss_u, test_loss,
    T1, T2, val_period, log):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/Planetoid/' + current_time + '/train'
    val_log_dir = 'logs/Planetoid/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    for epoch in range(1, epochs+1):

        print("Epoch: {:d} ==> ".format(epoch), end=' ')

        model.train_step(L_s, L_u, optimizer_u, optimizer_s, train_accuracy, train_loss, train_loss_u, T1, T2)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss_s', train_loss.result(), step=epoch)
            # tf.summary.scalar('loss_u', train_loss_u.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        if epoch % log == 0:
            print("Train Loss: s {:.3f} u {:.3f}, Train Accuracy: {:.2f}%".format(train_loss.result(), train_loss_u.result(), train_accuracy.result()*100))

        if epoch % val_period == 0:
            
            model.test_step(L_s, test_accuracy, test_loss, mode="val")

            with val_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            
            print("\nEpoch {:d}, Validation Loss: {:.3f}, Validation Accuracy: {:.2f}%\n".format(epoch, test_loss.result(), test_accuracy.result()*100))

        # Reset metrics every epoch
        train_loss.reset_states()
        # train_loss_u.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()


def test(model, L_s, test_accuracy, test_loss):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = 'logs/Planetoid/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model.test_step(L_s, test_accuracy, test_loss, mode="test")

    print("Test Loss: {:.3f}, Test Accuracy: {:.2f}%".format(test_loss.result(), test_accuracy.result()*100))

    return


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", default=5/6)
    parser.add_argument("--r2", default=5/6)
    parser.add_argument("--q",  default=10)
    parser.add_argument("--d",  default=3)
    parser.add_argument("--n1", default=200)
    parser.add_argument("--n2", default=200)
    parser.add_argument("--t1", default=20)
    parser.add_argument("--t2", default=20)  
 
    args = parser.parse_args()


    embedding_size = 50    
    dataset = "cora"    # "cora" "pubmed" "citeseer"
    seed = 1234
    
    modality = "T"          # can be T (transductive) or I (inductive)    
    
    epochs = 10
    val_period = 5        # each epoch validation
    log = 1                 # every two epochs print train loss and acc
    pre_train_iters = 100    # graph context pretrain iterations

    # args = {'r1': 5/6, 'r2': 5/6, 'q':10 , 'd':3, 'n1':200, 'n2':200, 't1':20, 't2':20}

    print("Planetoid-{:s}!".format(modality))

    # Preprocess on data
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    permute(features, neighbors, labels, o_h_labels, keys, seed)
    train_idx, val_idx, test_idx = split(dataset, len(features))

    # Define model, loss, metrics and optimizers
    if modality == "I":
        model = Planetoid_I(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)
    elif modality == "T":
        model = Planetoid_T(
            features, neighbors, o_h_labels, embedding_size, train_idx, val_idx, test_idx, args)

    L_s = tf.keras.losses.CategoricalCrossentropy()
    L_u = UnlabeldLoss(args.n2)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_acc")
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_u = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    optimizer_u = tf.keras.optimizers.SGD(learning_rate=1e-2)       # , momentum=0.99)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=1e-1)       # , momentum=0.99)


    # Pretrain iterations on graph context
    model.pretrain_step(L_u, optimizer_u, pre_train_iters)

    # Train model    
    train(model, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, test_accuracy, train_loss, train_loss_u, test_loss,
         args.t1, args.t2, val_period, log)

    # Test model 
    test(model, L_s, test_accuracy, test_loss)
