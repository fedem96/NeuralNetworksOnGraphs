import tensorflow as tf
import numpy as np

# TODO: sparse implementation
class Chebychev(tf.keras.layers.Layer):

    # Chebychev Spectral Convolutional Layer
    # T(k, L): chebychev polinomial of order k of laplacian matrix L
    # T(0, L) = I
    # T(1, L) = 2L
    # T(k, L) = 2L * T(k-1, L) - T(k-2, L)
    # computes: sum([theta[k] * T(k, L) * x  for k in K])

    def __init__(self, laplacian, max_order, num_filters):
        super().__init__()
        print("init")

        self.bilaplacian = tf.cast( tf.constant(2 * laplacian), tf.float16)
        self.max_order = max_order
        self.n = len(self.bilaplacian)
        self.f = -1
        self.num_filters = num_filters

        std = 0.1 
        self.theta =  tf.Variable(tf.random.normal([num_filters, max_order+1], stddev=std, dtype=tf.float16), dtype=tf.float16, name="theta")
        self.polynomials = self._chebychev_polynomials()

    def _chebychev_polynomials(self):

        # 0-order
        polys = tf.cast(tf.eye(self.n), tf.float16)

        # 1-order
        if self.max_order >= 1:
            polys = tf.stack([polys, self.bilaplacian])

        for k in range(2, self.max_order+1):
            # k-order
            poly_k = tf.matmul(self.bilaplacian, polys[k-1]) - polys[k-2]
            poly_k = tf.expand_dims(poly_k, 0)
            polys = tf.concat([polys, poly_k], 0)

        return tf.reshape(polys, [self.max_order+1, self.n*self.n])

    def build(self, input_shape):
        print("build")
        print("input_shape:", input_shape)
        assert len(input_shape) == 2
        self.f = input_shape[1]

    def call(self, x):
        print("call")
        print("x:", x)
        x = tf.cast(x, tf.float16)
        x = tf.reshape(x, [self.n, self.f])
        tp = tf.matmul(self.theta, self.polynomials)
        tp = tf.reshape(tp, [self.num_filters, self.n, self.n])
        o = tf.matmul(tp, x)
        o = tf.transpose(o, [1, 0, 2])
        o = tf.reshape(o, [self.n, self.num_filters * self.f])
        return o
