import tensorflow as tf


class UnlabeledLoss(tf.keras.losses.Loss):

    def __init__(self, N2):
        super().__init__()
        self.N2 = N2

    def call(self, y_true, y_pred):
        s = tf.reduce_sum(y_pred, axis=1)
        dot_prod = tf.math.multiply(s, y_true)
        # Credits to https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        # loss = -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))
        loss = tf.reduce_sum(tf.nn.softplus(-dot_prod))
        return loss
