import abc
import six

import tensorflow as tf

from vary import tensor_utils as tensor_utils

@six.add_metaclass(abc.ABCMeta)
class _Flow(object):
    """Single iteration of a normalizing flow"""
    def __init__(self, n_latent_dim, random_state=123):
        self.random_state = random_state
        self.n_latent_dim = n_latent_dim
        self.built_ = False
        self.build()

    def build(self):
        self._built = True

    @abc.abstractmethod
    def transform(self, z_sample):
        pass

    @abc.abstractmethod
    def log_det_jacobian(self, z_sample):
        pass


class _VolumePreservingFlow(_Flow):
    """Volume preserving flow."""
    def log_det_jacobian(self, z_sample):
        batch_size = tf.shape(z_sample)[0]
        return tf.zeros([batch_size])


class _IdentityFlow(_VolumePreservingFlow):
    def transform(self, z_sample):
        return z_sample


@six.add_metaclass(abc.ABCMeta)
class NormalizingFlow(object):
    def __init__(self, name, n_iter=2, random_state=123):
        self.name = name
        self.n_iter = n_iter
        self.random_state = random_state
        self._built = False

    @abc.abstractproperty
    def flow_class(self):
        pass

    def build(self, n_latent_dim):
        if not self._built:
            self._flows = [self.flow_class(n_latent_dim, random_state=self.random_state)
                           for _ in range(self.n_iter)]
        self._built = True

    def transform(self, z_sample):
        with tf.variable_scope('normalizing_flow_transform',
                               self.name + '_transform',
                               [z_sample]):
            z_sample = tensor_utils.to_tensor(z_sample, dtype=tf.float32)
            n_latent_dim = tensor_utils.get_shape(z_sample)[1]
            self.build(n_latent_dim)
            for flow in self._flows:
                z_sample = flow.transform(z_sample)
            return z_sample

    def log_det_jacobian(self, z_sample):
        with tf.variable_scope('normalizing_flow_log_det_jacobian',
                               self.name + '_log_det_jacobian',
                               [z_sample]):
            z_sample = tensor_utils.to_tensor(z_sample, dtype=tf.float32)
            n_latent_dim = tensor_utils.get_shape(z_sample)[1]
            self.build(n_latent_dim)
            log_det_jacobians = []
            for flow in self._flows:
                log_det_jacobians.append(flow.log_det_jacobian(z_sample))
                z_sample = flow.transform(z_sample)

            return tf.add_n(log_det_jacobians)
