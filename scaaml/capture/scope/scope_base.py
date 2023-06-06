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
"""Context manager for the scope."""

from abc import ABC, abstractmethod
from typing import Optional, Union

from chipwhisperer.capture.scopes import OpenADC

from scaaml.capture.scope.scope_template import ScopeTemplate


class AbstractSScope(ABC):
    """Scope context manager."""

    def __init__(self, samples: int, offset: int):
        """Create scope context.

        Args:
          samples: How samples to capture (length of the capture).
          offset: Number of samples to wait after trigger event occurred before
            starting recording data.
        """
        self._scope: Optional[Union[OpenADC, ScopeTemplate]] = None
        self._samples: int = samples
        self._offset: int = offset

    @abstractmethod
    def __enter__(self):
        """Create scope context.

        Returns: self
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """

    @property
    def scope(self) -> Optional[Union[OpenADC, ScopeTemplate]]:
        """Scope object for chipwhisperer API."""
        return self._scope
