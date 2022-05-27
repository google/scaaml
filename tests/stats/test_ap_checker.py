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
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from scaaml.stats import APChecker


@patch.object(APChecker, '_run_check')
def test_run_all_calls_check_all_nonzero(mock_run_check):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)

    mock_run_check.assert_called_with(APChecker.check_all_nonzero)


@patch.object(APChecker, 'run_all')
def test_init_calls_run_all(mock_run_all):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)

    mock_run_all.assert_called_once_with()


@patch.object(APChecker, 'check_all_nonzero')
def test_run_all_calls_check_all_nonzero(mock_check_all_nonzero):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)

    mock_check_all_nonzero.assert_called_once_with()


def test_attack_point_name():
    attack_point_name = MagicMock()
    counts = np.array([])

    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)

    assert ap_checker.attack_point_name == attack_point_name


def test_check_all_nonzero():
    attack_point_name = 'some_strange_attack_p0int_name'
    counts = np.array([[1, 2, 3], [2, 3, 1]])

    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)
    assert not ap_checker._something_failed

    counts[1][1] = 0
    ap_checker = APChecker(counts=counts, attack_point_name=attack_point_name)
    assert ap_checker._something_failed
