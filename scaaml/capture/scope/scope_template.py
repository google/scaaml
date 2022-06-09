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
"""Base class for a scope handle. An object that can be passed as a scope to
chipwhisperer API."""

from abc import abstractmethod

import numpy as np

from chipwhisperer.capture.api.cwcommon import ChipWhispererCommonInterface


class ScopeTemplate(ChipWhispererCommonInterface):
    """A base class for scope objects that can be passed as a scope to
    chipwhisperer API (such as Pico6424E)."""

    def __init__(self):
        """Initialize the base."""

    @abstractmethod
    def con(self):
        """Connect to the attached hardware."""

    @abstractmethod
    def dis(self):
        """Disconnect."""

    @abstractmethod
    def arm(self):
        """Setup scope to begin capture when triggered."""

    @abstractmethod
    def capture(self):
        """Capture trace (must be armed first)."""

    @abstractmethod
    def get_last_trace(self, as_int: bool = False) -> np.ndarray:
        """Return the last trace."""

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of this object."""
