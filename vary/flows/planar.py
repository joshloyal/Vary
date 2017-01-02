"""Planar Normalizing Flow"""
import tensorflow as tf
from tensorflow.contrib import slim

from vary.flows.registry import RegisterFlow
from vary.flows.base import NormalizingFlow
from vary.flows.base import _Flow

from vary import tensor_utils as tensor_utils
from vary import ops

class _PlanarFlow(_Flow):
    def build(self)
        """Calculate a vector u_hat that ensure invertibility (appendix A.1)"""
        self.planar_U = tf.get_variable('planar_U',
                                        shape=[self.n_latent_dim, 1],
                                        initializer=None,
                                        dtype=tf.float32,
                                        trainable=True)

        self.planar_W = tf.get_variable('planar_W',
                                        shape=[self.n_latent_dim, 1],
                                        initializer=None,
                                        dtype=tf.float32,
                                        trainable=True)

        self.planar_bias = tf.get_variable('planar_bias',
                                           shape=[],
                                           initializer=tf.zeros_initializer(),
                                           dtype=tf.float32,
                                           trainable=True)

        UW = ops.vector_dot(self.planar_U, self.planar_W)
        mUW= -1 + tf.nn.softplus(UW)  # -1 + log(1 + exp(uw))
        norm_W = self.planar_W / tf.reduce_sum(self.planar_W ** 2)
        self.U_hat = self.planar_U + (mUW - UW) * norm_W

        super(_PlanarFlow, self).build()

    def transform(self, z_sample, features=None):
        with tf.variable_scope('planar_flow', [z_sample]):
            z_hat = tf.xw_plus_b(z_sample, self.planar_W, self.planar_bias)  # shape [batch_size, 1]

            f_z = z_sample + self.U_hat * tf.squeeze(tf.nn.tanh(z_hat)) # [n_latent_dim, 1] * [batch_size]

            # psi = h'(w^T * z + b) * w
            # where h' = tanh' = 1 - tanh**2
            psi = tf.matmul(1 - tf.nn.tanh(z_hat) ** 2, self.planar_W)

            # log_det_jac = log(|1 + u_hat^T * psi|)
            log_det_jac = tf.log(tf.abs(1 + tf.matmul(psi, self.U_hat)))

            return f_z, log_det_jac


@RegisterFlow('planar')
class PlanarFlow(NormalizingFlow):
    def __init__(self, n_iter=2, random_state=123):
        super(Planar, self).__init__(
            name='planar_flow',
            n_iter=n_iter,
            random_state=random_state)

    @property
    def flow_class(self):
        return _PlanarFlow


#@RegisterFlow('inverse_autoregressive')
#class InverseAutoRegressiveFlow(NormalizingFlow):
#    pass
