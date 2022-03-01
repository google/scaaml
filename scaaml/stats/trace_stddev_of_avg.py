"""Compute stddev of averages of traces."""

from typing import List

import numpy as np

from scaaml.stats.trace_statistic_base import AbstractTraceStatistic


class STDDEVofAVGofTraces(AbstractTraceStatistic):
    """Computes standard deviation of averages of traces.

    Example use:
      stddev_avgs = STDDEVofAVGofTraces()
      for e in ExampleIterator(ds_path):
          stddev_avgs.update(e['trace1'])
      print(stddev_avgs.result())
    """
    def __init__(self) -> None:
        super().__init__()
        # List of averages.
        self._averages: List[float] = []

    def update(self, trace: np.ndarray) -> None:
        """Update the statistic with average of a single trace.

        Args:
          trace: Numpy one dimensional array of floats.
        """
        self._averages.append(trace.mean())

    def result(self) -> float:
        """Return the standard deviation of all the averages.

        Raises: If STDDEVofAVGofTraces.update has never been called,
        RuntimeWarning is raised (computing standard deviation of an empty
        array).
        """
        averages = np.array(self._averages, dtype=np.float64)
        return averages.std()
