import numpy as np
import pytest
import tensorflow as tf
from tensorflow.contrib import layers

from vary import tensor_utils
from vary import ops
from vary.flows import IdentityFlow, HouseHolderFlow


module_rng = np.random.RandomState(123)


class TestFlows(tf.test.TestCase):
    def test_identity_flow(self):
        flow = IdentityFlow(n_iter=1)

        z_sample = module_rng.randn(10, 5)
        with self.test_session() as sess:
            trans = flow.transform(z_sample)
            self.assertAllClose(z_sample, trans.eval())

            jacobian = flow.log_det_jacobian(z_sample)
            self.assertAllClose(np.zeros(10), jacobian.eval())

    def test_householder_no_input(self):
        flow = HouseHolderFlow(n_iter=1)

        z_sample = module_rng.randn(10, 5)
        with pytest.raises(Exception):
            trans = flow.transform(z_sample)

    def test_householder_flow(self):
        flow = HouseHolderFlow(n_iter=2)

        z_sample = module_rng.randn(10, 5)
        z_input = layers.utils.collect_named_outputs(
            ops.VariationalParams.COLLECTION,
            ops.VariationalParams.INPUT,
            tensor_utils.to_tensor(
                module_rng.randn(10, 30), dtype=tf.float32)
        )
        with self.test_session() as sess:
            trans_1 = flow.transform(z_sample)
            trans_2 = flow.transform(z_sample)
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(trans_1.eval(), trans_2.eval())

            jacobian = flow.log_det_jacobian(z_sample)
            self.assertAllClose(np.zeros(10), jacobian.eval())
