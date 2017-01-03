import numpy as np
from scipy import linalg
import tensorflow as tf
import tensorflow.contrib.learn as learn
from keras.datasets import mnist
from keras.utils import np_utils

from vary import GaussianVAE

tf.logging.set_verbosity(tf.logging.INFO)

def build_simple_topics():
    n_topics = 3
    block = n_topics * np.ones((3, 3))
    blocks = [block] * n_topics
    X = linalg.block_diag(*blocks)
    labels = np.eye(3)[np.repeat([0, 1, 2], 3)]
    return n_topics, X.astype(np.float32), labels


class TestGaussianVAE(tf.test.TestCase):
    def test_gaussian_vae(self):
        n_topics, X, _ = build_simple_topics()
        vae = GaussianVAE(n_latent_dim=3, n_iter=1, n_jobs=-1)
        vae.fit(X)
        print(vae.transform(X))

    def test_householder_flow(self):
        n_topics, X, _ = build_simple_topics()
        vae = GaussianVAE(n_latent_dim=3, n_iter=1,
                          normalizing_flow='householder',
                          n_jobs=-1)
        vae.fit(X)
        print(vae.transform(X))

    def test_planar_flow(self):
        n_topics, X, _ = build_simple_topics()
        vae = GaussianVAE(n_latent_dim=3, n_iter=1,
                          normalizing_flow='planar',
                          n_jobs=-1)
        vae.fit(X)
        print(vae.transform(X))

    #def test_mnist(self):
    #    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #    X_train = X_train.reshape(60000, 784)
    #    X_test = X_test.reshape(10000, 784)
    #    X_train = X_train.astype('float32')
    #    X_test = X_test.astype('float32')
    #    X_train /= 255
    #    X_test /= 255

    #    model_op = vae.topic_vae(n_latent_dim=2)
    #    config = learn.RunConfig(num_cores=8, tf_random_seed=123)
    #    estimator = learn.Estimator(model_fn=model_op, config=config)

    #    estimator.fit(X_train, steps=1000, batch_size=32)
    #    print(estimator.predict(X_train))
