# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The target in cw."""
import chipwhisperer as cw
from chipwhisperer.capture.targets.SimpleSerial import SimpleSerial

from scaaml.capture.communication import AbstractSCommunication


class CWCommunication(AbstractSCommunication):
    """target in cw"""

    def __init__(self, scope):
        """Initialize the communication object."""
        super().__init__()
        self._scope = scope
        self._target = None
        self._protver = '1.1'

    def __enter__(self):
        """Initialize target."""
        assert self._target is None  # Do not allow nested with.
        # The scope is there because of communication with the target (it
        # communicated using single USB endpoint). Since CW 5.5 firmware
        # release it uses a separate USB UART.
        self._target = cw.target(self._scope, cw.targets.SimpleSerial)
        assert type(self._target) == SimpleSerial
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
        assert self._target is not None
        self._target.dis()
        self._target = None

    @property
    def target(self):
        """Returns the target object."""
        return self._target
