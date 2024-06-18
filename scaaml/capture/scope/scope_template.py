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
"""Base class for a scope handle. An object that can be passed as a scope to
chipwhisperer API."""

from typing import Optional, Protocol, Union

import numpy as np
import numpy.typing as npt

ScopeTraceType = npt.NDArray[np.float32]
ScopeDigitalTraceType = npt.NDArray[np.bool_]
ScopeTriggerTraceType = Union[ScopeTraceType, ScopeDigitalTraceType]


class ScopeTemplate(Protocol):
    """A base class for scope objects that can be passed as a scope to
    chipwhisperer API (such as Pico6424E)."""

    def con(self, sn: Optional[str] = None) -> bool:
        """Connect to the attached hardware. Trying to keep compatibility with
        cw.capture.scopes.OpenADC and being able to pass as `scope` argument to
        `cw.capture_trace`.

        Returns: True if the connection was successful, False otherwise.
        """

    def dis(self) -> bool:
        """Disconnect.

        Returns: True if the disconnection was successful, False otherwise.
        """

    def arm(self) -> None:
        """Setup scope to begin capture when triggered."""

    def capture(self, poll_done: bool = False) -> bool:
        """Capture trace (must be armed first). Same signature as
        cw.capture.scopes.OpenADC.

        Returns: True if the capture timed out, False if it did not.
        """

    def get_last_trace(self, as_int: bool = False) -> ScopeTraceType:
        """Return the last trace. Same signature as cw.capture.scopes.OpenADC.
        """

    def __str__(self) -> str:
        """Return string representation of this object."""
