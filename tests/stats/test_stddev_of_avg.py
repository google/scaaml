import os
from unittest.mock import MagicMock, patch

import numpy as np

from scaaml.stats import STDDEVofAVGofTraces

def test_init():
    stddev_of_avg = STDDEVofAVGofTraces()
    assert len(stddev_of_avg._averages) == 0


def test_update():
    stddev_of_avg = STDDEVofAVGofTraces()

    stddev_of_avg.update(np.array([1, 2, 3]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([2.0, 2.0, 2.0]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([4.0, 5.0, 6.0]))
    assert np.isclose(stddev_of_avg.result(), 1.4142135623730951)
