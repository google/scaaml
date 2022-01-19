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


class AbstractSScope(ABC):
    """Scope context manager."""
    def __init__(self, samples: int, offset: int, **_):
        """Create scope context.

        Args:
          samples: How samples to capture (length of the capture).
          offset: Number of samples to wait after trigger event occurred before
            starting recording data.
          _: CWScope is expected to be initialized using the capture_info
            dictionary which may contain extra keys (additional information
            about the capture; the capture_info dictionary is saved in the
            info file of the dataset). Thus we can ignore the rest of keyword
            arguments.

        Expected use:
          capture_info = {
              'samples': samples,
              'offset': offset,
              'other_information': 'Can also be present.',
          }
          with CWScope(**capture_info) as scope:
              # Use the scope object.
        """
        self._scope = None
        self._samples = samples
        self._offset = offset

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
