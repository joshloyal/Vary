import numpy as np
import tensorflow as tf
from vary import tensor_utils

def subtril(subtril_values, n_dim):
    subtril_values = tensor_utils.to_tensor(subtril_values, dtype=tf.float32)

    indices = list(zip(*np.tril_indices(n_dim, -1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    return tf.sparse_to_dense(sparse_indices=indices, output_shape=[n_dim, n_dim],
                              sparse_values=subtril_values, default_value=0,
                              validate_indices=True)


def reshape_cholskey(diag_values, subtril_values):
    diag_values = tensor_utils.to_tensor(diag_values, dtype=tf.float32)
    subtril_values = tensor_utils.to_tensor(subtril_values, dtype=tf.float32)
    n_dim = tensor_utils.get_shape(diag_values)[1]

    return tf.diag(diag_values) + subtril(subtril_values, n_dim=n_dim)


array = np.arange(25).reshape(5, 5)
subtril_values = array[np.tril_indices_from(array, -1)]
diag_values = np.diag(array)
print(array)
print

subtril_values = np.vstack((subtril_values, subtril_values))
diag_values = np.vstack((diag_values, diag_values))
with tf.Session() as sess:
    print sess.run(reshape_cholskey(diag_values, subtril_values))
