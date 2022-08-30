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
"""Compute a statistic of many traces."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractTraceStatistic(ABC):
    """Base class to compute a statistic over many traces.

    Example use:
      trace_stats = ChildClass()
      for e in ExampleIterator(ds_path):
          trace_stats.update(e['trace1'])
      print(trace_stats.result())
    """

    def __init__(self) -> None:
        """Create a new statistic computation."""

    @abstractmethod
    def update(self, trace: np.ndarray) -> None:
        """Update the statistic with a single trace.

        Args:
          trace: Numpy one dimensional array of floats.
        """

    @abstractmethod
    def result(self):
        """Return the statistic."""
