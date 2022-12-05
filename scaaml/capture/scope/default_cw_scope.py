# Copyright 2022 Google LLC
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

from typing import Optional, Union

import chipwhisperer as cw
from chipwhisperer.capture.scopes.OpenADC import OpenADC
from chipwhisperer.capture.scopes.CWNano import CWNano

from scaaml.capture.scope import AbstractSScope


class DefaultCWScope(AbstractSScope):
    """Scope context manager."""

    def __init__(self):
        """Create scope context."""
        super().__init__(samples=0, offset=0)
        self._scope: Optional[Union[OpenADC, CWNano]] = None

    def __enter__(self):
        """Create scope context.

        Returns: self
        """
        assert self._scope is None  # Do not allow nested with.

        # Open cw scope with default settings.
        self._scope = cw.scope()
        assert self._scope is not None
        self._scope.default_setup()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
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
