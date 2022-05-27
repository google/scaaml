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

from scaaml.stats import APCounter


def test_init():
    attack_point_info = {
        'len': 16,
        'max_val': 256,
    }

    ap_counter = APCounter(attack_point_info=attack_point_info)

    assert ap_counter._len == attack_point_info['len']
    assert ap_counter._max_val == attack_point_info['max_val']
    assert (ap_counter._cnt == 0).all()
    assert ap_counter._cnt.shape == (ap_counter._len, ap_counter._max_val)
    assert ap_counter._cnt.dtype == np.int64


def test_get_counts():
    attack_point_info = {
        'len': 15,
        'max_val': 259,
    }

    ap_counter = APCounter(attack_point_info=attack_point_info)

    counts = ap_counter.get_counts()
    assert (counts == 0).all()
    assert counts.shape == (ap_counter._len, ap_counter._max_val)
    assert counts.dtype == np.int64

    counts_b = ap_counter.get_counts(byte=1)
    assert (counts_b == 0).all()
    assert counts_b.shape == (ap_counter._max_val,)
    assert counts_b.dtype == np.int64


def test_update():
    attack_point_info = {
        'len': 7,
        'max_val': 13,
    }

    ap_counter = APCounter(attack_point_info=attack_point_info)
    ap_counter.update([0, 1, 2, 12, 4, 5, 6])

    counts = ap_counter.get_counts()
    assert counts[0][0] == 1
    assert counts[1][1] == 1
    assert counts[2][2] == 1
    assert counts[3][12] == 1
    assert counts[4][4] == 1
    assert counts[5][5] == 1
    assert counts[6][6] == 1
    assert counts.sum() == 7
    assert (counts >= 0).all()

    count_b = ap_counter.get_counts(byte=1)
    assert count_b[1] == 1
    assert count_b.sum() == 1

    ap_counter.update([1, 7, 3, 12, 5, 4, 6])
    assert counts[0][0] == 1
    assert counts[0][1] == 1
    assert counts[1][1] == 1
    assert counts[1][7] == 1
    assert counts[2][2] == 1
    assert counts[2][3] == 1
    assert counts[3][12] == 2
    assert counts[4][4] == 1
    assert counts[4][5] == 1
    assert counts[5][5] == 1
    assert counts[5][4] == 1
    assert counts[6][6] == 2
    assert counts.sum() == 14
    assert (counts >= 0).all()

    count_b = ap_counter.get_counts(byte=1)
    assert count_b[1] == 1
    assert count_b[7] == 1
    assert count_b.sum() == 2
