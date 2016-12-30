"""Normalizing Flows"""
import tensorflow as tf
from tensorflow.contrib import slim

from vary import tensor_utils as tensor_utils


def householder_flow_single(z_sample,
                            z_input,
                            random_state=123,
                            name=None):
    """Implements a single iteration of the Householder transformation.

    Parameters
    ----------
    z_sample : tensor of shape [batch_size, n_latent_dim]
        The sample generated from the previous iteration of the transformation.

    z_input : tensor of shape [batch_size, n_variational_params]
        The output of the fully-connected network used as input
        to construct the parameter's of the initial variational distribution.

    random_state : int
        Seed to the random number generator

    name : str
        Name of the operation

    Returns
    -------
    z_new : tensor of shape [batch_size, n_latent_dim]
        Result of the transformation
    """
    with tf.variable_scope('householder_flow_single', name,
                           [z_sample, z_input]):
        z_sample = tensor_utils.to_tensor(z_sample, dtype=tf.float32)
        z_input = tensor_utils.to_tensor(z_input, dtype=tf.float32)
        n_latent_dim = tensor_utils.get_shape(z_sample)[1]

        v = slim.fully_connected(z_input,
                                 num_outputs=n_latent_dim,
                                 activation_fn=None)
        v_norm = tf.nn.l2_normalize(v, dim=[1])

        v_1 = tf.tile(tf.expand_dims(v_norm, 2), [1, 1, n_latent_dim])
        v_2 = tf.tile(tf.expand_dims(v_norm, 1), [1, n_latent_dim, 1])

        z_new = (z_sample -
                 tf.reduce_sum(
                    2 * (v_1 * v_2) * tf.expand_dims(z_sample, -1), 2))

        return z_new


def householder_flow(z_sample,
                     z_input,
                     n_iter=2,
                     random_state=123,
                     name=None):
    """Implements the Householder Transformation, which is a
    valume-preserving normalizing flow. The purpose of narmalizing flows
    is to improve a VAE's evidence lower bound by performing a series
    of invertible transformations of latent variables with simple posteriors
    to generate a final random variable with a more flexible posterior.

    The evidence lower bound is given by

        E_q[ ln(p(x|z_T) + sum(ln(det(Jac(F)))) ] - KL(q(z_0|x)||p(z_T))

    where the sum is taken from t = 1 ... T and Jac(F) is the Jacobian
    matrix of the normalizing flow transformation wrt the latent variable.

    The householder transformation is a special normalizing flow that is also
    volume preserving, i.e. has zero Jacobian determinant. This means that
    performing this transformation results in a posterior with an
    (approximately) full-covariance matrix, while leaving the objective
    unmodified.

    Parameters
    ----------
    z_sample : tensor of shape [batch_size, n_latent_dim]
        A sample from the initial latent distribution

    z_input : tensor of shape [batch_size, n_variational_params]
        The output of the fully-connected network used as input
        to construct the parameter's of the initial variational distribution.

    random_state : int
        Seed to the random number generator

    name : str
        Name of the operation

    Returns
    -------
    q_z_k : tensor of shape [batch_size, n_latent_dim]
        A sample from the posterior of the distribution obtained
        by applying the householder transformation.

    Reference
    ---------
    - Jakub M. Tomczak and Max Welling,
      "Improving Variational Auto-Encoders using Householder Flow".
    """
    with tf.variable_scope('householder_flow', name, [z_sample, z_input]):
        for k in range(n_iter):
            z_sample = householder_flow_single(
                z_sample,
                z_input,
                random_state=random_state,
                name='householder_flow_iteration_{}'.format(k))

            return z_sample
