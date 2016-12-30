import numpy as np
import tensorflow as tf
from tensorflow.contrib.bayesflow import stochastic_tensor as st


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
