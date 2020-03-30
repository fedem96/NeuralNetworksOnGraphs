import tensorflow as tf
import numpy as np
import sys
import os
import datetime
import argparse


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
            tf.summary.scalar('loss_u', train_loss_u.result(), step=epoch)
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
        train_loss_u.reset_states()
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

