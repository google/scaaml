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
