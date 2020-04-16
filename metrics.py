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

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_pred, axis=1)
        dot_prod = tf.math.multiply(s, tf.squeeze(y_true))
        loss = - tf.reduce_sum(tf.math.log(tf.sigmoid(dot_prod)))
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

class EarlyStoppingAvg(tf.keras.callbacks.Callback):
    """Stop training only when loss does not improve over the mean of a fixed-size sliding window

    Arguments:
        patience: size of the sliding window (number of epochs)
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', restore_best_weights=False, baseline=None, from_epoch=0):
        super(EarlyStoppingAvg, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.from_epoch = from_epoch

        if mode == 'auto':
            if 'acc' in monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'
        else:
            assert mode in ['min', 'max']
            self.mode = mode

        if self.mode == 'max' and baseline is not None: self.baseline = -self.baseline

    def on_train_begin(self, logs=None):
        self.window = []
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = None
        self.baseline_reached = False

    def on_epoch_end(self, epoch, logs=None):
        self.last = logs.get(self.monitor)
        if self.mode == 'max': self.last = -self.last

        if self.baseline is None or self.last < self.baseline:
            self.baseline_reached = True

        if self.window == []:
            self.window = [self.last]
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        out_of_patience = self.patience <= 0 or len(self.window) == self.patience

        if epoch >= self.from_epoch and self.baseline_reached and self.last > np.mean(self.window) + self.min_delta and out_of_patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch
        else:
            self.window.append(self.last)
            if len(self.window) > self.patience:
                self.window.pop(0)

        if self.last < self.best:
            self.best = self.last
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            self.model.set_weights(self.best_weights)

        if self.stopped_epoch > 0:
            if self.mode == 'max': self.last = -self.last
            if self.mode == 'max': self.best = -self.best
            print('Early stop at epoch {:d} with best {} {:.3f}, last {:.3f}' .format(self.stopped_epoch, self.monitor, self.best, self.last))