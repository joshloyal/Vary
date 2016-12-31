import abc
import six

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.contrib import learn

from vary.parallel import get_n_jobs


def _steps_per_iter(X, batch_size):
    n_samples = X.shape[0]
    return int(np.ceil(n_samples / batch_size))


@six.add_metaclass(abc.ABCMeta)
class BaseTensorFlowModel(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_iter=10,
                 learning_rate=1e-3,
                 optimizer='Adam',
                 batch_size=32,
                 n_jobs=1,
                 random_state=123):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state

    @abc.abstractmethod
    def _model_spec(self):
        pass

    def _train_op(self, objective):
        train_op = tf.contrib.layers.optimize_loss(
            objective,
            tf.contrib.framework.get_global_step(),
            optimizer=self.optimizer,
            learning_rate=self.learning_rate)

        return train_op

    def fit(self, X, y=None):
        config = learn.RunConfig(
            num_cores=get_n_jobs(self.n_jobs),
            tf_random_seed=self.random_state)

        self.model = learn.Estimator(
            model_fn=self._model_spec(),
            config=config)

        steps_per_iter = _steps_per_iter(X, self.batch_size)
        n_steps = int(steps_per_iter * self.n_iter)

        self.model.fit(X, y=y, steps=n_steps, batch_size=self.batch_size)

    def transform(self, X):
        return self.model.predict(X)
