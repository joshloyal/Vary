import numpy as np
import pytest
import tensorflow as tf
from tensorflow.contrib import layers

from vary import tensor_utils
from vary import ops
from vary import enums
from vary import flows

module_rng = np.random.RandomState(123)


class TestFlows(tf.test.TestCase):
    def test_identity_flow(self):
        flow = flows.IdentityFlow(n_iter=1)

        z_sample = module_rng.randn(10, 5)
        with self.test_session() as sess:
            trans, jacobian = flow.transform(z_sample)
            self.assertAllClose(z_sample, trans.eval())
            self.assertAllClose(np.zeros(10), jacobian.eval())

    def test_householder_no_input(self):
        flow = flows.HouseHolderFlow(n_iter=1)

        z_sample = module_rng.randn(10, 5)
        with pytest.raises(Exception):
            trans, jacobian = flow.transform(z_sample)

    def test_householder_flow(self):
        flow = flows.HouseHolderFlow(n_iter=2)

        features = module_rng.randn(10, 30)
        z_sample = module_rng.randn(10, 5)
        with self.test_session() as sess:
            trans_1, jac_1 = flow.transform(z_sample, features)
            trans_2, jac_2 = flow.transform(z_sample, features)
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(trans_1.eval(), trans_2.eval())
            self.assertAllClose(np.zeros(10), jac_1.eval())
            self.assertAllClose(jac_1.eval(), jac_2.eval())

    def test_planar_flow(self):
        flow = flows.PlanarFlow(n_iter=1)

        z_sample = module_rng.randn(10, 5)
        with self.test_session() as sess:
            trans, jac = flow.transform(z_sample)
            sess.run(tf.global_variables_initializer())
            print(trans.eval())
            print(jac.eval())
