import tensorflow as tf
import numpy as np


def train(model, x_train, y_train, x_val, y_val, epochs, optimizer,
        loss_fn, train_loss, train_acc, val_loss, val_acc, val_period):

    for epoch in range(epochs+1):
        
        print("Epoch: {:d} ==> ".format(epoch), end=' ')

        with tf.GradientTape() as tape:
            predictions = model(x_train)
            loss = loss_fn(y_train, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss(loss)
        train_acc(y_train, predictions)

        print("Train Loss: {:.3f}, Train Accuracy: {:.2f}%".format(train_loss.result(), train_acc.result()*100))

        if epoch % val_period == 0:

            predictions = model(x_val)
            loss = loss_fn(y_val, predictions)

            val_loss(loss)
            val_acc(y_val, predictions)

            print("\nEpoch {:d}, Validation Loss: {:.3f}, Validation Accuracy: {:.2f}%\n".format(epoch, val_loss.result(), val_acc.result()*100))
 
    return


def test():

    return
