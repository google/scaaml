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

import os
from unittest.mock import MagicMock, patch

import numpy as np

from scaaml.stats import STDDEVofSTATofTraces
from scaaml.stats import STDDEVofAVGofTraces
from scaaml.stats import STDDEVofMAXofTraces
from scaaml.stats import STDDEVofMINofTraces


def test_stddev_of_stat():
    stat_fn = MagicMock()
    side_effects = [MagicMock() for _ in range(4)]
    stat_fn.side_effect = side_effects
    stddev_of_stat = STDDEVofSTATofTraces(stat_fn=stat_fn)

    assert stddev_of_stat._stats == []
    assert not stat_fn.called

    for i in range(len(side_effects)):
        mm_arg = MagicMock()
        stddev_of_stat.update(mm_arg)
        stat_fn.assert_called_with(mm_arg)
        assert stddev_of_stat._stats == side_effects[:i + 1]


def test_std_of_avg():
    stddev_of_avg = STDDEVofAVGofTraces()

    stddev_of_avg.update(np.array([1, 2, 3]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([2.0, 2.0, 2.0]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([4.0, 5.0, 6.0]))
    assert np.isclose(stddev_of_avg.result(), 1.4142135623730951)


def test_std_of_max():
    stddev_of_avg = STDDEVofMAXofTraces()

    stddev_of_avg.update(np.array([1, 2, 3]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([2.0, 2.0, 2.0]))
    assert stddev_of_avg.result() == 0.5
    stddev_of_avg.update(np.array([4.0, 5.0, 6.0]))
    assert np.isclose(stddev_of_avg.result(), 1.699673171197595)


def test_std_of_min():
    stddev_of_avg = STDDEVofMINofTraces()

    stddev_of_avg.update(np.array([1, 2, 3]))
    assert stddev_of_avg.result() == 0.0
    stddev_of_avg.update(np.array([2.0, 2.0, 2.0]))
    assert stddev_of_avg.result() == 0.5
    stddev_of_avg.update(np.array([4.0, 5.0, 6.0]))
    assert np.isclose(stddev_of_avg.result(), 1.247219128924647)
