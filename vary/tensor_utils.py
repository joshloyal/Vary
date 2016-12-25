import tensorflow as tf

def to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def get_shape(tensor):
    """Get numeric shape value"""
    if isinstance(tensor, tf.Tensor):
        return tensor.get_shape().as_list()
    elif isinstance(tensor, (np.array, list, tuple)):
        return np.shape(tensor)
