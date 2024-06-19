# Copyright 2021-2024 Google LLC
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
from types import TracebackType
from typing import Optional
from typing_extensions import Self

import chipwhisperer as cw
from chipwhisperer.capture.targets import TargetTypes
from chipwhisperer.capture.targets.SimpleSerial import SimpleSerial

from scaaml.capture.communication import AbstractSCommunication
from scaaml.capture.scope.scope_template import ScopeTemplate


class CWCommunication(AbstractSCommunication):
    """target in cw"""

    def __init__(self, scope: Optional[ScopeTemplate]) -> None:
        """Initialize the communication object."""
        super().__init__()
        self._scope: Optional[ScopeTemplate] = scope
        self._target: Optional[TargetTypes] = None
        self._protver = '1.1'

    def __enter__(self) -> Self:
        """Initialize target."""
        assert self._target is None  # Do not allow nested with.
        # The scope is there because of communication with the target (it
        # communicated using single USB endpoint). Since CW 5.5 firmware
        # release it uses a separate USB UART.
        # self._scope here is a CWNano, we know it is a ScopeTemplate, but cw
        # typing wants OpenADC | CWNano | None. We don't want to check
        # isinstance because we use a mock in unit-tests.
        self._target = cw.target(
            self._scope,  # type: ignore[arg-type]
            SimpleSerial,
        )
        if isinstance(self._target, SimpleSerial):
            self._target.protver = self._protver
        self._scope = None
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        assert self._target is not None
        self._target.dis()  # type: ignore[no-untyped-call]
        self._target = None

    @property
    def target(self) -> Optional[TargetTypes]:
        """Returns the target object."""
        return self._target
