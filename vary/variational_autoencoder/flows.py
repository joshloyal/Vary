"""Normalizing Flows"""
import tensorflow as tf
from tensorflow.contrib import slim

from vary import tensor_utils as tensor_utils


def householder_flow_single(z_state,
                            z_input,
                            random_state=123,
                            name=None):
    with tf.variable_scope('householder_flow_single', name,
                           [z_state, z_input]):
        z_state = tensor_utils.to_tensor(z_state, dtype=tf.float32)
        z_input = tensor_utils.to_tensor(z_input, dtype=tf.float32)
        n_latent_dim = tensor_utils.get_shape(z_state)[1]

        v = slim.fully_connected(z_input,
                                 num_outputs=n_latent_dim,
                                 activation_fn=None)
        v_norm = tf.nn.l2_normalize(v, dim=[1])

        v_1 = tf.tile(tf.expand_dims(v_norm, 2), [1, 1, n_latent_dim])
        v_2 = tf.tile(tf.expand_dims(v_norm, 1), [1, n_latent_dim, 1])

        z_new = (z_state -
                 tf.reduce_sum(
                    2 * (v_1 * v_2) * tf.expand_dims(z_state, -1), 2))

        return z_new


def householder_flow(z_state,
                     z_input,
                     n_iter=2,
                     random_state=123,
                     name=None):
    with tf.variable_scope('householder_flow', name, [z_state, z_input]):
        for k in range(n_iter):
            z_state = householder_flow_single(
                z_state,
                z_input,
                random_state=random_state,
                name='householder_flow_iteration_{}'.format(k))

            return z_state
