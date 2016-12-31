"""Normalizing Flows"""
import abc
import six

import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.layers as layers

from vary import ops
from vary import tensor_utils as tensor_utils


@six.add_metaclass(abc.ABCMeta)
class NormalizingFlow(object):
    def __init__(self, name, n_iter=2, random_state=123):
        self.name = name
        self.n_iter = n_iter
        self.random_state = random_state

    @abc.abstractmethod
    def _transform(self, z_sample):
        pass

    def transform(self, z_sample):
        with tf.variable_scope('normalizing_flow', self.name, [z_sample]):
            for k in range(self.n_iter):
                z_sample = self._transform(z_sample)
            return z_sample


class VolumePreservingFlowMixin(object):
    """Volume preserving flow."""
    def log_det_jacobian(self, z_sample):
        batch_size = tf.shape(z_sample)[0]
        return tf.zeros([batch_size])


_NORMALIZING_FLOWS = {}


class RegisterFlow(object):
    """Decorator to registor NormalizingFlow classes.

    Parameters
    ----------
    flow_cls_name : str
        name of the NormalizingFlow class. This will be used in the
        `get_flow` method as the key for the class.

    Usage
    -----
    >>> @flow_lib.RegisterFlow('MyFlow')
    >>> class MyFlow(NormalizingFlow)
    >>>     ...
    """
    def __init__(self, flow_cls_name):
        self._key = flow_cls_name

    def __call__(self, flow_cls):
        if not hasattr(flow_cls, 'transform'):
            raise TypeError("flow_cls must implement a `transform` method, "
                            "recieved %s" % flow_cls)

        if not hasattr(flow_cls, 'log_det_jacobian'):
            raise TypeError("flow_cls must implement a `log_det_jacobian` "
                            "method, recieved %s" % flow_cls)

        if self._key in _NORMALIZING_FLOWS:
            raise ValueError("%s has already been registered to : %s"
                             % (self._key, _NORMALIZING_FLOWS[self._key]))

        _NORMALIZING_FLOWS[self._key] = flow_cls
        return flow_cls


def _registered_flow(name):
    """Get the normalizing flow class registered to `name`."""
    return _NORMALIZING_FLOWS.get(name, None)


def get_flow(name, n_iter=2, random_state=123):
    flow_class = _registered_flow(name)
    if flow_class is None:
        raise NotImplementedError(
            "No Normalizing Flow registered with name %s" % name)

    return flow_class(n_iter=n_iter, random_state=random_state)


@RegisterFlow('identity')
class IdentityFlow(NormalizingFlow, VolumePreservingFlowMixin):
    """No-op for consistency."""
    def __init__(self, n_iter=2, random_state=123):
        super(IdentityFlow, self).__init__(
            name='identity_flow',
            n_iter=n_iter,
            random_state=random_state)

    def _transform(self, z_sample):
        return z_sample


def householder_flow_single(z_sample,
                            random_state=123,
                            name=None):
    """Implements a single iteration of the Householder transformation.

    Parameters
    ----------
    z_sample : tensor of shape [batch_size, n_latent_dim]
        The sample generated from the previous iteration of the transformation.

    random_state : int
        Seed to the random number generator

    name : str
        Name of the operation

    Returns
    -------
    z_new : tensor of shape [batch_size, n_latent_dim]
        Result of the transformation
    """
    with tf.variable_scope('householder_flow_single', name, [z_sample]):
        z_sample = tensor_utils.to_tensor(z_sample, dtype=tf.float32)
        n_latent_dim = tensor_utils.get_shape(z_sample)[1]

        # z_input : tensor of shape [batch_size, n_variational_params]
        #    The output of the fully-connected network used as input
        #    to construct the parameter's of the initial variational
        #    distribution.
        z_input = layers.utils.convert_collection_to_dict(
            ops.VariationalParams.COLLECTION)[ops.VariationalParams.INPUT]

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


@RegisterFlow('householder')
class HouseholderFlow(NormalizingFlow, VolumePreservingFlowMixin):
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
    n_iter : int
        Number of iterations to perform the normalizing flow.

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
    def __init__(self, n_iter=2, random_state=123):
        super(HouseholderFlow, self).__init__(
            name='householder_flow',
            n_iter=n_iter,
            random_state=random_state)

    def _transform(self, z_sample):
        return householder_flow_single(z_sample,
                                       random_state=self.random_state)


def planar_flow_variables(n_latent_dim):
    """Calculate a vector u_hat that ensure invertibility (appendix A.1)"""
    planar_u = tf.get_variable('planar_u',
                               shape=[n_latent_dim],
                               initializer=None,
                               dtype=tf.float32,
                               trainable=True)

    planar_w = tf.get_variable('planar_w',
                               shape=[n_latent_dim],
                               initializer=None,
                               dtype=tf.float32,
                               trainable=True)

    bias = tf.get_variable('planar_bias',
                           shape=[],
                           initializer=tf.zeros_initializer(),
                           dtype=tf.float32,
                           trainable=True)

    uw = tf.matmul(planar_u, planar_w)
    muw = -1 + tf.nn.softplus(uw)  # -1 + log(1 + exp(uw))
    u_hat = (planar_u +
             (muw - uw) * tf.transpose(planar_w) /
                tf.reduce_sum(planar_w ** 2))

    return u_hat, planar_w, bias


def planar_flow(z_sample,
                name=None):
    with tf.variable_scope('planar_flow', name, [z_sample]):
        n_latent_dim = tensor_utils.get_shape(z_sample)[1]
        U, W, bias = planar_flow_variables(n_latent_dim)
        z_hat = tf.xw_plus_b(z_sample, W, bias)


#@RegisterFlow('planar')
#class PlanarFlow(NormalizingFlow):
#    pass
#
#
#@RegisterFlow('inverse_autoregressive')
#class InverseAutoRegressiveFlow(NormalizingFlow):
#    pass
