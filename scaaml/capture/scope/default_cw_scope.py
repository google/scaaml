# Copyright 2022-2024 Google LLC
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
"""Context manager chipwhisperer scope that has default setting and is used to
control."""

from types import TracebackType
from typing import Optional, Type
from typing_extensions import Self

import chipwhisperer as cw
from chipwhisperer.capture.scopes.cwnano import CWNano

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.scope_template import ScopeTemplate


class DefaultCWScope(AbstractSScope):
    """Scope context manager."""

    def __init__(self, cw_scope_serial_number: Optional[str] = None):
        """Create scope context.

        Args:
          cw_scope_serial_number (Optional[str]): Serial number is needed when
            more scopes are connected at the same time. Default to None
            (connect to the single scope). Just the number, without the device
            name and colon character.
        """
        super().__init__(samples=0, offset=0)
        self._scope: Optional[ScopeTemplate] = None
        self._cw_scope_serial_number: Optional[str] = cw_scope_serial_number

    def __enter__(self) -> Self:
        """Create scope context.

        Returns: self
        """
        assert self._scope is None  # Do not allow nested with.

        # Open cw scope with default settings.

        scope = cw.scope(sn=self._cw_scope_serial_number)
        assert not isinstance(scope, CWNano)
        self._scope = scope

        assert self._scope is not None
        self._scope.default_setup()  # type: ignore[no-untyped-call]
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        if self._scope is None:
            return
        self._scope.dis()
        self._scope = None
