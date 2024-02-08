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
"""Communication protocols between the capturing host and the chip."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional
from typing_extensions import Self

from chipwhisperer.capture.targets import TargetTypes


class AbstractSCommunication(ABC):
    """Base class for communication. Communication is a context manager."""

    def __init__(self) -> None:
        """Initialization of the communication object."""

    @abstractmethod
    def __enter__(self) -> Self:
        """Initialize target."""
        return self

    @abstractmethod
    def __exit__(self, exc_type: type[BaseException],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """

    @property
    def target(self) -> Optional[TargetTypes]:
        """The target object. Returns None if no target is available."""
        return None
