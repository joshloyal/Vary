"""Variational Autoencoder"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.contrib.bayesflow import variational_inference as vi

from vary.base import BaseNVIModel
from vary import tensor_utils as tensor_utils
from vary.variational_autoencoder.ops import gaussian_inference_network, generative_network


def variational_autoencoder(features, n_latent_dim=2, hidden_units=[500, 500], kl_weight=1.0):
    features = tensor_utils.to_tensor(features, dtype=tf.float32)
    kl_weight = tensor_utils.to_tensor(kl_weight, dtype=tf.float32)

    n_features = features.get_shape().as_list()[1]
    with tf.variable_scope('inference_network'):
        q_mu, q_sigma = gaussian_inference_network(x=features,
                                                   n_latent_dim=n_latent_dim,
                                                   hidden_units=hidden_units)

    # set up the latent variables
    with tf.variable_scope('latent_samples'):
        with st.value_type(st.SampleValue(1)):
            q_z = st.StochasticTensor(distributions.Normal, mu=q_mu, sigma=q_sigma)

    # set up the priors
    with tf.variable_scope('prior'):
        prior = distributions.Normal(mu=np.zeros(n_latent_dim, dtype=np.float32),
                                     sigma=np.ones(n_latent_dim, dtype=np.float32))
        vi.register_prior(q_z, prior)

    with tf.variable_scope('generative_network'):
        p_x_given_z_logits = generative_network(z=q_z,
                                                hidden_units=hidden_units,
                                                n_features=n_features)
        p_x_given_z = distributions.Bernoulli(logits=p_x_given_z_logits)

    # set up the elbo
    log_likelihood = tf.reduce_sum(p_x_given_z.log_pmf(features), 1)
    log_likelihood = tf.expand_dims(log_likelihood, -1)
    elbo = vi.elbo(log_likelihood)
    objective = tf.reduce_mean(-elbo)

    #kl = tf.reduce_sum(distributions.kl(q_z.distribution, p_z), 1)
    #expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_pmf(features), 1)
    #elbo = tf.reduce_sum(expected_log_likelihood - kl_weight * kl, 0)

    train_op = tf.contrib.layers.optimize_loss(
        objective, tf.contrib.framework.get_global_step(), optimizer='Adam',
        learning_rate=1e-3)

    return q_mu, objective, train_op


def vae_model(n_latent_dim=2, hidden_units=[500, 500]):
    def model_spec(features, labels=None):
        return variational_autoencoder(features, n_latent_dim, hidden_units)
    return model_spec


class GaussianVAE(BaseNVIModel):
    def __init__(self, n_latent_dim=2, hidden_units=[500, 500], kl_weight=1.0,
                 n_iter=10, batch_size=32, n_jobs=1, random_state=123):
        self.n_latent_dim = n_latent_dim
        self.hidden_units = hidden_units
        self.kl_weight = kl_weight

        super(GaussianVAE, self).__init__(
            n_iter=n_iter,
            batch_size=batch_size,
            n_jobs=n_jobs,
            random_state=random_state)

    def _model_spec(self):
        return vae_model(
            n_latent_dim=self.n_latent_dim,
            hidden_units=self.hidden_units)
