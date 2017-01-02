import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.bayesflow import stochastic_tensor as st

from vary.exceptions import GraphLookupError
from vary import enums

def to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def get_shape(tensor):
    """Get numeric shape value"""
    if isinstance(tensor, tf.Tensor):
        return tensor.get_shape().as_list()
    elif isinstance(tensor, st.StochasticTensor):
        return tensor.value().get_shape().as_list()
    elif isinstance(tensor, (np.ndarray, list, tuple)):
        return np.shape(tensor)


def fetch_variational_inputs():
    """Lookup the inputs to the variational parameters network. This
    is the layer before variational parameters network."""
    try:
        z_input = layers.utils.convert_collection_to_dict(
            enums.VariationalParams.COLLECTION)[enums.VariationalParams.INPUT]
    except KeyError:
        raise GraphLookupError('Could not fetch variational parameters. '
                               'Please construct a generative distribution '
                               'using one of the ops in `vary.ops`.')

    return z_input
