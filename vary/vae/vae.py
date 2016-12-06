import abc
import six

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
from tensorflow.contrib.bayesflow import stochastic_tensor as st

from vary.base import BaseNVIModel


def inference_network(x, n_latent_dim, hidden_units):
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


def generative_network(z, hidden_units, n_features):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.stack(z, slim.fully_connected, hidden_units,
                         scope='decoder_network')
        bernoulli_logits = slim.fully_connected(net, n_features,
                                                activation_fn=None)
        return bernoulli_logits


def model_op(X, n_latent_dim=2, hidden_units=[500, 500]):
    X = tf.convert_to_tensor(X)
    X = tf.cast(X, tf.float32)

    n_features = X.get_shape().as_list()[1]
    with tf.variable_scope('inference_network'):
        q_mu, q_sigma = inference_network(x=X,
                                          n_latent_dim=n_latent_dim,
                                          hidden_units=hidden_units)

    with tf.variable_scope('latent_samples'):
        with st.value_type(st.SampleAndReshapeValue()):
            q_z = st.StochasticTensor(distributions.Normal, mu=q_mu, sigma=q_sigma)

    with tf.variable_scope('generative_network'):
        p_x_given_z_logits = generative_network(z=q_z,
                                                hidden_units=hidden_units,
                                                n_features=n_features)
        p_x_given_z = distributions.Bernoulli(logits=p_x_given_z_logits)

    with tf.variable_scope('prior'):
        p_z = distributions.Normal(mu=np.zeros(n_latent_dim, dtype=np.float32),
                                   sigma=np.ones(n_latent_dim, dtype=np.float32))

    kl = tf.reduce_sum(distributions.kl(q_z.distribution, p_z), 1)
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_pmf(X), 1)
    elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)

    train_op = tf.contrib.layers.optimize_loss(
        -elbo, tf.contrib.framework.get_global_step(), optimizer='Adam',
        learning_rate=1e-3)

    return q_mu, -elbo, train_op


def topic_vae(n_latent_dim=2, hidden_units=[500, 500]):
    def model_spec(features, labels=None):
        return model_op(features, n_latent_dim, hidden_units)
    return model_spec


@six.add_metaclass(abc.ABCMeta)
class GaussianVAE(BaseNVIModel):
    def __init__(self, n_latent_dim=2, hidden_units=[500, 500], n_iter=10,
                 batch_size=32, n_jobs=1, random_state=123):
        self.n_latent_dim = n_latent_dim
        self.hidden_units = hidden_units

        super(GaussianVAE, self).__init__(
            n_iter=n_iter,
            batch_size=batch_size,
            n_jobs=n_jobs,
            random_state=random_state)

    def _model_spec(self):
        return topic_vae(
            n_latent_dim=self.n_latent_dim,
            hidden_units=self.hidden_units)
