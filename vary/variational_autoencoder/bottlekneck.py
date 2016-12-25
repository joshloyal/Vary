"""Variational Information Bottlekneck"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.contrib.bayesflow import variational_inference as vi

from vary.base import BaseNVIModel
from vary import tensor_utils as tensor_utils
from vary.variational_autoencoder.ops import mvn_inference_network, logistic_regression


def variational_information_bottlekneck(features,
                                        labels,
                                        n_latent_dim=2,
                                        hidden_units=[500, 500],
                                        full_covariance=False,
                                        kl_weight=1.0):
    features = tensor_utils.to_tensor(features, dtype=tf.float32)
    kl_weight = tensor_utils.to_tensor(kl_weight, dtype=tf.float32)

    n_features = tensor_utils.get_shape(features)[1]
    n_classes = tensor_utils.get_shape(labels)[1]
    with tf.variable_scope('inference_network'):
        q_mu, q_chol = mvn_inference_network(x=features,
                                             n_latent_dim=n_latent_dim,
                                             hidden_units=hidden_units)

    with tf.variable_scope('latent_samples'):
        with st.value_type(st.SampleValue(1)):
            q_z = st.StochasticTensor(distributions.MultivariateNormalCholesky,
                                      mu=q_mu,
                                      chol=q_chol)

    with tf.variable_scope('prior'):
        #p_z = distributions.MultivariateNormalDiag(
        #    mu=np.zeros(n_latent_dim, dtype=np.float32),
        #    diag_stdev=np.ones(n_latent_dim, dtype=np.float32))
        # need to make mu and chol constants of same shape as q_z [n_samples, n_latent_dim, n_latent_dim]
        prior = distributions.MultivariateNormalCholesky(
            mu=np.zeros(n_latent_dim, dtype=np.float32).reshape(-1, n_latent_dim),
            chol=np.eye(n_latent_dim, dtype=np.float32).reshape(-1, n_latent_dim, n_latent_dim))
        vi.register_prior(q_z, prior)

    with tf.variable_scope('discrimative_network'):
        p_y_logits = logistic_regression(features=q_z,
                                         n_outputs=n_classes)


    log_likelihood = -tf.nn.softmax_cross_entropy_with_logits(p_y_logits, labels)
    log_likelihood = tf.expand_dims(log_likelihood, -1)
    elbo = vi.elbo(log_likelihood)
    objective = tf.reduce_mean(-elbo)
    #kl = tf.reduce_sum(distributions.kl(q_z.distribution, p_z), 1)
    #expected_nll = tf.nn.softmax_cross_entropy_with_logits(p_y_logits, labels)
    #objective = tf.reduce_sum(expected_nll + kl_weight * kl, 0)

    train_op = tf.contrib.layers.optimize_loss(
        objective, tf.contrib.framework.get_global_step(), optimizer='Adam',
        learning_rate=1e-3)

    return q_mu, objective, train_op


def vib_model(n_latent_dim=2, hidden_units=[500, 500]):
    def model_spec(features, labels=None):
        return variational_information_bottlekneck(features, labels, n_latent_dim, hidden_units)
    return model_spec


class InformationBottlekneck(BaseNVIModel):
    def __init__(self, n_latent_dim=2, hidden_units=[500, 500], kl_weight=1.0,
                 n_iter=10, batch_size=32, n_jobs=1, random_state=123):
        self.n_latent_dim = n_latent_dim
        self.hidden_units = hidden_units
        self.kl_weight = kl_weight

        super(InformationBottlekneck, self).__init__(
            n_iter=n_iter,
            batch_size=batch_size,
            n_jobs=n_jobs,
            random_state=random_state)

    def _model_spec(self):
        return vib_model(
            n_latent_dim=self.n_latent_dim,
            hidden_units=self.hidden_units)
