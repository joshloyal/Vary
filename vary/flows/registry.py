_NORMALIZING_FLOWS = {}


class RegisterFlow(object):
    """Decorator to registor NormalizingFlow classes.

    Parameters
    ----------
    flow_cls_name : str
        name of the NormalizingFlow class. This will be used in the
        `get_flow` method as the key for the class.

    Usage
    -----
    >>> @flow_lib.RegisterFlow('MyFlow')
    >>> class MyFlow(NormalizingFlow)
    >>>     ...
    """
    def __init__(self, flow_cls_name):
        self._key = flow_cls_name

    def __call__(self, flow_cls):
        if not hasattr(flow_cls, 'transform'):
            raise TypeError("flow_cls must implement a `transform` method, "
                            "recieved %s" % flow_cls)

        if self._key in _NORMALIZING_FLOWS:
            raise ValueError("%s has already been registered to : %s"
                             % (self._key, _NORMALIZING_FLOWS[self._key]))

        _NORMALIZING_FLOWS[self._key] = flow_cls
        return flow_cls


def _registered_flow(name):
    """Get the normalizing flow class registered to `name`."""
    return _NORMALIZING_FLOWS.get(name, None)


def get_flow(name, n_iter=2, random_state=123):
    flow_class = _registered_flow(name)
    if flow_class is None:
        raise NotImplementedError(
            "No Normalizing Flow registered with name %s" % name)

    return flow_class(n_iter=n_iter, random_state=random_state)
