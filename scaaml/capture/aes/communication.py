"""The target in cw."""
import chipwhisperer as cw


class SCommunication:
    """target in cw"""
    def __init__(self, scope):
        self._scope = scope
        self._target = None
        self._protver = '1.1'

    def __enter__(self):
        """Initialize target."""
        assert self._target is None  # Do not allow nested with.
        self._target = cw.target(self._scope, cw.targets.SimpleSerial)
        self._target.protver = self._protver
        self._scope = None
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        self._target.dis()
        self._target = None

    @property
    def target(self):
        """Returns the target object."""
        return self._target
