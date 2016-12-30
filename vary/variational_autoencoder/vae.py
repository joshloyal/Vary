"""Variational Autoencoder"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
from tensorflow.contrib.bayesflow import stochastic_tensor as st

from vary.base import BaseTensorFlowModel
from vary.variational_autoencoder import flows
from vary import tensor_utils as tensor_utils
from vary import ops


def variational_autoencoder(features,
                            n_latent_dim=2,
                            hidden_units=[500, 500],
                            normalizing_flow='identity',
                            flow_n_iter=2,
                            kl_weight=1.0,
                            random_state=123):
    features = tensor_utils.to_tensor(features, dtype=tf.float32)
    kl_weight = tensor_utils.to_tensor(kl_weight, dtype=tf.float32)

    n_features = tensor_utils.get_shape(features)[1]
    with tf.variable_scope('inference_network'):
        q_mu, q_sigma = ops.gaussian_inference_network(x=features,
                                                       n_latent_dim=n_latent_dim,
                                                       hidden_units=hidden_units)
        #q_mu, q_chol = ops.mvn_inference_network(x=features,
        #                                         n_latent_dim=n_latent_dim,
        #                                         hidden_units=hidden_units)
    # set up the latent variables
    with tf.variable_scope('latent_samples'):
        with st.value_type(st.SampleValue()):
            q_z = st.StochasticTensor(
                dist=distributions.Normal(mu=q_mu, sigma=q_sigma),
                name='q_z')
            #q_z = st.StochasticTensor(
            #    dist=distributions.MultivariateNormalCholesky(
            #        mu=q_mu, chol=q_chol),
            #        name='q_z')

        z_input = layers.utils.convert_collection_to_dict(
            ops.VariationalParams.COLLECTION)[ops.VariationalParams.INPUT]

        norm_flow = flows.get_flow(normalizing_flow,
                                   n_iter=flow_n_iter,
                                   random_state=random_state)

    # set up the priors
    with tf.variable_scope('prior'):
        prior = distributions.Normal(
            mu=np.zeros(n_latent_dim, dtype=np.float32),
            sigma=np.ones(n_latent_dim, dtype=np.float32))

    with tf.variable_scope('generative_network'):
        p_x_given_z = ops.bernoulli_generative_network(
            z=norm_flow.transform(q_z, z_input),
            hidden_units=hidden_units,
            n_features=n_features)

    # set up elbo
    log_likelihood = tf.reduce_sum(p_x_given_z.log_pmf(features), 1)
    log_det_jac = norm_flow.log_det_jacobian(q_z)
    kl = tf.reduce_sum(distributions.kl(q_z.distribution, prior), 1)
    neg_elbo = -tf.reduce_sum(log_likelihood + log_det_jac - kl_weight * kl, 0)

    return q_mu, tf.identity(neg_elbo, name='neg_elbo')


def vae_model(train_fn,
              n_latent_dim=2,
              hidden_units=[500, 500],
              normalizing_flow='identity',
              flow_n_iter=2):
    def model_spec(features, labels=None):
        q_mu, neg_elbo = variational_autoencoder(
            features, n_latent_dim, hidden_units,
            normalizing_flow, flow_n_iter)
        train_op = train_fn(neg_elbo)
        return q_mu, neg_elbo, train_op
    return model_spec


class GaussianVAE(BaseTensorFlowModel):
    def __init__(self,
                 n_latent_dim=2,
                 hidden_units=[500, 500],
                 normalizing_flow='identity',
                 flow_n_iter=2,
                 kl_weight=1.0,
                 n_iter=10,
                 learning_rate=1e-3,
                 optimizer='Adam',
                 batch_size=32,
                 n_jobs=1,
                 random_state=123):
        self.n_latent_dim = n_latent_dim
        self.hidden_units = hidden_units
        self.normalizing_flow = normalizing_flow
        self.flow_n_iter = flow_n_iter
        self.kl_weight = kl_weight

        super(GaussianVAE, self).__init__(
            n_iter=n_iter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            random_state=random_state)

    def _model_spec(self):
        return vae_model(
            self._train_op,
            n_latent_dim=self.n_latent_dim,
            hidden_units=self.hidden_units,
            normalizing_flow=self.normalizing_flow,
            flow_n_iter=self.flow_n_iter)
