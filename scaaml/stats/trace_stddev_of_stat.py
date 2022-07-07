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
"""Compute stddev of (avg, max, min) of traces."""

from typing import Callable, List

import numpy as np

from scaaml.stats.trace_statistic_base import AbstractTraceStatistic


class STDDEVofSTATofTraces(AbstractTraceStatistic):
    """Computes standard deviation of stats of traces.

    Example use: See STDDEVofAVGofTraces.
    """

    def __init__(self, stat_fn: Callable[[np.ndarray], float]) -> None:
        """Initialize empty statistic.

        Args:
          stat_fn: A function that takes a trace (one dimensional np array of
            floats) and returns a float.
        """
        super().__init__()
        self._stat_fn = stat_fn
        # List of stats.
        self._stats: List[float] = []

    def update(self, trace: np.ndarray) -> None:
        """Update with a single trace.

        Args:
          trace: Numpy one dimensional array of floats.
        """
        self._stats.append(self._stat_fn(trace))

    def result(self) -> float:
        """Return the standard deviation of all the stats.

        Raises: If STDDEVofAVGofTraces.update has never been called,
        RuntimeWarning is raised (computing standard deviation of an empty
        array).
        """
        stats = np.array(self._stats, dtype=np.float64)
        return stats.std()


class STDDEVofAVGofTraces(STDDEVofSTATofTraces):
    """Computes standard deviation of averages of traces.

    Example use:
      stddev_averages = STDDEVofAVGofTraces()
      for e in ExampleIterator(ds_path):
          stddev_averages.update(e['trace1'])
      print(stddev_averages.result())
    """

    def __init__(self) -> None:

        def stat_fn(x):
            return x.mean()

        super().__init__(stat_fn=stat_fn)


class STDDEVofMAXofTraces(STDDEVofSTATofTraces):
    """Computes standard deviation of maxima of traces."""

    def __init__(self) -> None:

        def stat_fn(x):
            return x.max()

        super().__init__(stat_fn=stat_fn)


class STDDEVofMINofTraces(STDDEVofSTATofTraces):
    """Computes standard deviation of minima of traces."""

    def __init__(self) -> None:

        def stat_fn(x):
            return x.min()

        super().__init__(stat_fn=stat_fn)
