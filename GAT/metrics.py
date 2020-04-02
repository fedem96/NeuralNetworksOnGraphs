import tensorflow as tf
import numpy as np
import os

class RegularizedLoss(tf.keras.losses.Loss):
    
    def __init__(self, l2_weight):
        super().__init__()
        self.l2_weight = l2_weight

    def call(self, y_true, y_pred):
        loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        l2_loss = tf.nn.l2_loss(y_pred-y_true)
        return loss + self.l2_weight * l2_loss


class EarlyStop(tf.keras.callbacks.Callback):

    def __init__(self, model, monitor, patience=0, save_model=False, checkpoint_path='./', verbose=1):
        super(EarlyStop, self).__init__()
        self.model = model
        self.monitor = monitor
        self.modality = 'acc' if 'acc' in monitor.name else 'loss'
        self.value_to_track = monitor.name.replace('_'+self.modality, '')
        self.patience = patience
        self.best_weights = None
        self.save_model = save_model
        self.stop_training = False
        if self.save_model:
            ckpt_name = self.model.name + 'ckpts/cp-{epoch:03d}-' + self.modality + '-{v:03f}.ckpt'
            self.checkpoint_path = os.path.join(checkpoint_path, ckpt_name)
        self.verbose = verbose

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf if self.modality == 'loss' else 0

    def on_epoch_end(self, epoch):
        current = self.monitor.result()
        if (np.greater(current, self.best) and self.modality == 'acc') or (np.less(current, self.best) and self.modality == 'loss'):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            if self.save_model:
                self.model.save_weights(self.checkpoint_path.format(
                    epoch=epoch, v=self.best))
                if self.verbose == 1:
                    print('Checkpoint saved with {} {:03f}'.format(self.monitor.name, self.best))
        else:
            self.wait += 1
            if self.wait == self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch {:05d}: early stopping for {}_{}' .format (self.stopped_epoch, self.value_to_track, self.modality))
    
    def get_status(self):
        return self.stop_training