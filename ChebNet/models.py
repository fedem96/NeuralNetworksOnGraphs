import tensorflow as tf
import numpy as np
from tqdm import tqdm

from layers import Chebychev

class ChebNet(tf.keras.models.Sequential):

    def __init__(self, norm_L, K, num_classes):
        super().__init__([
            tf.keras.layers.Dropout(0.5),
            Chebychev(norm_L, K, num_filters=16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            Chebychev(norm_L, K, num_filters=num_classes, activation="softmax"),
        ])
        self.normalized_laplacian = norm_L
        self.K = K

        self.compile(
            loss=self.non_zero_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[ChebNet.accuracy_mask]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.get('categorical_crossentropy')
        self.l2_weight = 5e-4


    def train(self, X, y, epochs):
        for epoch in tqdm(range(epochs)):
            loss, y_pred = self.train_step(X, y)
            tqdm.write("epoch: {}/{}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, epochs, loss, float(ChebNet.accuracy_mask(y, y_pred))))

    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            y_pred = self.call(X)                 # forward pass
            loss = self.non_zero_loss(y, y_pred)  # calculate loss
            
        grads = tape.gradient(loss, self.trainable_weights)                # backpropagation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) # optimizer step
        return loss, y_pred

    def non_zero_loss(self, y_true, y_pred):
        mask = ChebNet.non_zero_mask(y_true)
        indices = tf.where(mask)
        classification_loss = tf.reduce_mean(self.loss_fn(tf.gather(y_true, indices), tf.gather(y_pred, indices)))
        l2_regularizer = tf.nn.l2_loss(self.trainable_weights[0]) # first layer only
        return classification_loss + self.l2_weight * l2_regularizer
        

    @staticmethod
    def non_zero_mask(y):
        return tf.reduce_sum(y, axis=1) != 0

    @staticmethod
    def accuracy_mask(y_true, y_pred):
        mask = ChebNet.non_zero_mask(y_true)
        pred_classes = tf.argmax(y_pred[mask], axis=1)
        true_classes = tf.argmax(y_true[mask], axis=1)
        return tf.reduce_sum( tf.cast(pred_classes == true_classes, tf.float32), keepdims=True ) / tf.reduce_sum( tf.cast(mask, tf.float32), keepdims=True )
