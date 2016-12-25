from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions


def gaussian_inference_network(x, n_latent_dim, hidden_units):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        hidden_units + [None]
        net = slim.stack(x, slim.fully_connected, hidden_units,
                         scope='encoder_network')

        gaussian_params = slim.fully_connected(
            net,
            2 * n_latent_dim,
            activation_fn=None,
            scope='gaussian_params')

    # The mean parameter is unconstrained
    mu = layers.utils.collect_named_outputs(
        'variational_params',
        'gaussian_mean',
        gaussian_params[:, :n_latent_dim])

    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    sigma = layers.utils.collect_named_outputs(
        'variational_params',
        'gaussian_sigma',
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

        mvn_params = slim.fully_connected(
            net,
            n_latent_dim * (n_latent_dim + 1),  # over parameterized
            activation_fn=None,
            scope='gaussian_params')

    # The mean parameter is unconstrained
    mu = mvn_params[:, :n_latent_dim]

    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    #sigma_diag = (
    #    1e-6 + tf.nn.softplus(mvn_params[:, n_latent_dim:2*n_latent_dim]))

    #sigma_subtril = mvn_params[:, n_laten_dim + 2*n_latent_dim:]

    # we can actually use fill_lower_triangular in TF 12 here
    chol = tf.reshape(mvn_params[:, n_latent_dim:], (-1, n_latent_dim, n_latent_dim))
    chol = distributions.matrix_diag_transform(chol, transform=tf.nn.softplus)

    return mu, chol


def generative_network(z, hidden_units, n_features):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.stack(z, slim.fully_connected, hidden_units,
                         scope='decoder_network')
        bernoulli_logits = slim.fully_connected(net, n_features,
                                                activation_fn=None)
        return bernoulli_logits


def logistic_regression(features, n_outputs):
    logits = slim.fully_connected(features, n_outputs, activation_fn=None)
    return logits

