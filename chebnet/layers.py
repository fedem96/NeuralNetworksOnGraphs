import tensorflow as tf
import numpy as np
import scipy

class Chebychev(tf.keras.layers.Layer):

    # Chebychev Spectral Convolutional Layer
    # T(k, L): chebychev polinomial of order k of scaled normalized laplacian matrix L
    # T(0, L) = I
    # T(1, L) = L
    # T(k, L) = 2L * T(k-1, L) - T(k-2, L)
    # computes: sum([theta[k] * T(k, L) * x  for k in K])

    def __init__(self, laplacian, K, num_filters, activation):
        super().__init__()

        self._dtype = tf.float32

        self.laplacian = laplacian

        self.K = K                       # number of chebychev orders
        self.n = self.laplacian.shape[0] # number of nodes
        self.fin = -1                    # number of input features
        self.fout = num_filters          # number of output features
        self.polynomials = self._chebychev_polynomials()  # pre-calculate Chebychev polynomials from order 0 to K-1
        self.activation = tf.keras.activations.get(activation)

    def _chebychev_polynomials(self):
        # returns a SparseTensor of shape (K*n, n) representing the K Chebychev matrices of shape (n, n)

        # 0-order polynomial
        tmp_2 = scipy.sparse.eye(self.n)  # temporary value (used in recurrent formulation)
        polys = scipy.sparse.eye(self.n)  # polys will contain all Chebychev's polynomials (v-stacked)

        # 1-order polynomial
        if self.K > 1:
            tmp_1 = self.laplacian        # temporary value (used in recurrent formulation)
            polys = scipy.sparse.vstack([polys, self.laplacian])

        for k in range(2, self.K):
            # k-order polynomial
            poly_k = 2 * self.laplacian.dot(tmp_1) - tmp_2   # Chebychev recurrent formulation for fast filtering
            polys = scipy.sparse.vstack([polys, poly_k])

            # update auxiliary matrices
            tmp_2 = tmp_1
            tmp_1 = poly_k

        # return polynomials (transformed from scipy.sparse.csr_matrix to tf.sparse.SparseTensor)
        coo_polys = polys.tocoo()
        indices = np.mat([coo_polys.row, coo_polys.col]).transpose()
        return tf.cast( tf.sparse.SparseTensor(indices, coo_polys.data, coo_polys.shape), self._dtype )

    def build(self, input_shape):
        self.fin = input_shape[1]
        self.theta = self.add_weight(shape=[self.K, self.fin, self.fout], initializer='glorot_uniform', dtype=self._dtype)

    def call(self, x):
        x = tf.cast(x, self._dtype)        
        tx = tf.sparse.sparse_dense_matmul(self.polynomials, x) # shapes: (K*n, n)    *    (n, fin)    -> (K*n, fin)
        tx = tf.reshape(tx, [self.K, self.n, self.fin])         # shapes: (K*n, fin)                   -> (K, n, fin)
        o = tf.matmul(tf.cast(tx, self._dtype), self.theta)     # shapes: (K, n, fin) * (K, fin, fout) -> (K, n, fout)
        o = tf.reduce_sum(o, 0)                                 # shapes: (K, n, fout)                 -> (n, fout)
        return o if self.activation is None else self.activation(o)
