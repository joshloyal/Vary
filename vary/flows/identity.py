from vary.flows.registry import RegisterFlow
from vary.flows.base import NormalizingFlow
from vary.flows.base import _VolumePreservingFlow


class _IdentityFlow(_VolumePreservingFlow):
    def transform(self, z_sample, features=None):
        return z_sample, self.log_det_jacobian(z_sample)


@RegisterFlow('identity')
class IdentityFlow(NormalizingFlow):
    """No-op for consistency."""
    def __init__(self, n_iter=2, random_state=123):
        super(IdentityFlow, self).__init__(
            name='identity_flow',
            n_iter=n_iter,
            random_state=random_state)

    @property
    def flow_class(self):
        return _IdentityFlow
