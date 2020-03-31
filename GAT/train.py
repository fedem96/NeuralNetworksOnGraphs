import tensorflow as tf
import numpy as np

from tqdm import tqdm


def train(model, features, labels, train_mask, val_mask, epochs, optimizer, loss_fn,
        train_loss, train_acc, val_loss, val_acc, callbacks, val_period):

    for c in callbacks:
        c.on_train_begin()

    for epoch in tqdm(range(1, epochs+1)):
        
        print("Epoch: {:d} ==> ".format(epoch), end=' ')

        with tf.GradientTape() as tape:
            predictions = model(features)
            loss = loss_fn(labels[train_mask], predictions[train_mask])
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss(loss)
        train_acc(labels[train_mask], predictions[train_mask])

        tqdm.write("\repoch: {}/{}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, epochs, train_loss.result(), train_acc.result()))

        if epoch % val_period == 0:

            predictions = model(features)
            loss = loss_fn(labels[val_mask], predictions[val_mask])

            val_loss(loss)
            val_acc(labels[val_mask], predictions[val_mask])

            print("Validation loss: {:.4f}, accuracy: {:.4f}\n".format(epoch, val_loss.result(), val_acc.result()))
        
        for c in callbacks:
            c.on_epoch_end(epoch)

    for c in callbacks:
        c.on_train_end()

    return


def test():

    return
