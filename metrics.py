import tensorflow as tf

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
        # Credits to https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        # loss = -1/self.N2 * tf.reduce_sum(tf.math.log_sigmoid(dot_prod))
        loss = tf.reduce_mean(tf.nn.softplus(-dot_prod))
        return loss

