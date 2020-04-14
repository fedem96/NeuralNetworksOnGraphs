import tensorflow as tf
import numpy as np 
import os

def masked_loss(y_true, y_pred, loss_name='categorical_crossentropy'):
    loss_fn = tf.keras.losses.get(loss_name)
    mask = (tf.reduce_sum(y_true, axis=1) != 0)
    indices = tf.where(mask)
    return tf.reduce_mean(loss_fn(tf.gather(y_true, indices), tf.gather(y_pred, indices)))

def masked_accuracy(y_true, y_pred):
    mask = (tf.reduce_sum(y_true, axis=1) != 0)
    pred_classes = tf.argmax(y_pred[mask], axis=1)
    true_classes = tf.argmax(y_true[mask], axis=1)
    return tf.reduce_sum( tf.cast(pred_classes == true_classes, tf.float32), keepdims=True ) / tf.reduce_sum( tf.cast(mask, tf.float32), keepdims=True )
    

class UnlabeledLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_pred, axis=1)
        dot_prod = tf.math.multiply(s, y_true)
        loss = - 1.0/self.N2 * tf.reduce_sum(tf.math.log(tf.sigmoid(dot_prod)))
        return loss

class EarlyStoppingAccLoss(tf.keras.callbacks.Callback):
    """Stop training only when both loss and acc do not improve

    Arguments:
        patience: Number of epochs to wait before the stop.
    """

    def __init__(self, patience=0, monitor="loss_acc", checkpoint_path=None, model_name=''):
        super(EarlyStoppingAccLoss, self).__init__()

        self.patience = patience
        self.monitor_name = monitor
        self.best_weights = None
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path == None:
            ckpt_name = model_name + '_ckpts/cp.ckpt'
            self.checkpoint_path = os.path.join(self.checkpoint_path, ckpt_name)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_l = np.Inf
        self.best_a = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current_l = logs.get('val_loss')
        current_a = logs.get('val_masked_accuracy')
        if (current_a >= self.best_a and 'acc' in self.monitor_name) or (current_l <= self.best_l and 'loss' in self.monitor_name):
            self.best_weights = self.model.get_weights()
            if not self.checkpoint_path == None:
                self.model.save_weights(self.checkpoint_path)   
            self.best_l = min(self.best_l, current_l)
            self.best_a = max(self.best_a, current_a)
            self.wait = 0
        elif self.patience > 0:  # patience<0 means no early stopping
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0:
            print('Early stop at epoch {:d} with best val_acc {:.3f} val_loss {:.3f}' .format(self.stopped_epoch,
                                                                                       self.best_a, self.best_l))