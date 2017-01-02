from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
from tensorflow.contrib.distributions import distribution_util

from vary import enums


def vector_dot(vec_1, vec_2, name=None):
    """Vector dot product. The input vectors are assumed to be one dimensional."""
    with tf.variable_scope('vector_dot', name, [vec_1, vec_2]):
        #return tf.matmul(vec_1, tf.reshape(vec_2, [-1, 1]))
        #return tf.squeeze(tf.matmul(
        #    tf.expand_dims(vec_1, [0]), tf.expand_dims(vec_2, [1])))
        return tf.matmul(tf.reshape(vec_1, [1, -1]), vec_2)

def gaussian_inference_network(x, n_latent_dim, hidden_units):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        hidden_units + [None]
        net = slim.stack(x, slim.fully_connected, hidden_units,
                         scope='encoder_network')

        # input layer to the gaussian latent variable
        layers.utils.collect_named_outputs(
            enums.VariationalParams.COLLECTION,
            enums.VariationalParams.INPUT,
            net)

        gaussian_params = slim.fully_connected(
            net,
            2 * n_latent_dim,
            activation_fn=None,
            scope='gaussian_params')

    # The mean parameter is unconstrained
    mu = layers.utils.collect_named_outputs(
        enums.VariationalParams.COLLECTION,
        enums.VariationalParams.LOCATION,
        gaussian_params[:, :n_latent_dim])

    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    sigma = layers.utils.collect_named_outputs(
        enums.VariationalParams.COLLECTION,
        enums.VariationalParams.SCALE,
        1e-6 + tf.nn.softplus(gaussian_params[:, n_latent_dim:]))

    return mu, sigma


def mvn_inference_network(x, n_latent_dim, hidden_units):
    """Multi-Variate Normal parameterized cholskey
    sigma = chol(sigma) * chol(sigma)^T
    """
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        hidden_units + [None]
        net = slim.stack(x, slim.fully_connected, hidden_units,
                         scope='encoder_network')

        # input layer to the gaussian latent variable
        layers.utils.collect_named_outputs(
            enums.VariationalParams.COLLECTION,
            enums.VariationalParams.INPUT,
            net)

        mvn_params = slim.fully_connected(
            net,
            n_latent_dim * (n_latent_dim + 1),  # over parameterized
            activation_fn=None,
            scope='gaussian_params')

    # The mean parameter is unconstrained
    mu = mvn_params[:, :n_latent_dim]

    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    chol_subtril = (
        1e-6 + tf.nn.softplus(mvn_params[:, n_latent_dim:2*n_latent_dim]))
    chol = distribution_util.fill_lower_triangular(chol_subtril)

    return mu, chol


def bernoulli_generative_network(z, hidden_units, n_features):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.stack(z, slim.fully_connected, hidden_units,
                         scope='decoder_network')
        bernoulli_logits = slim.fully_connected(net, n_features,
                                                activation_fn=None)

        return distributions.Bernoulli(logits=bernoulli_logits)


def logistic_regression(features, n_outputs):
    logits = slim.fully_connected(features, n_outputs, activation_fn=None)
    return logits

