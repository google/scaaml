import os
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from scaaml.stats import APChecker


@patch.object(APChecker, 'check_all_nonzero')
def test_run_all_calls_check_all_nonzero(mock_check_all_nonzero):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts,
                           attack_point_name=attack_point_name)

    mock_check_all_nonzero.assert_called_once_with()


@patch.object(APChecker, 'run_all')
def test_init_calls_run_all(mock_run_all):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts,
                           attack_point_name=attack_point_name)

    mock_run_all.assert_called_once_with()


@patch.object(APChecker, 'check_all_nonzero')
def test_run_all_calls_check_all_nonzero(mock_check_all_nonzero):
    attack_point_name = 'k'
    counts = np.array([])

    ap_checker = APChecker(counts=counts,
                           attack_point_name=attack_point_name)

    mock_check_all_nonzero.assert_called_once_with()


def test_attack_point_name():
    attack_point_name = MagicMock()
    counts = np.array([])

    ap_checker = APChecker(counts=counts,
                           attack_point_name=attack_point_name)

    assert ap_checker.attack_point_name == attack_point_name


def test_check_all_nonzero():
    attack_point_name = 'some_strange_attack_p0int_name'
    counts = np.array([[1, 2, 3], [2, 3, 1]])

    ap_checker = APChecker(counts=counts,
                           attack_point_name=attack_point_name)

    counts[1][1] = 0
    with pytest.raises(ValueError) as verror:
        ap_checker = APChecker(counts=counts,
                               attack_point_name=attack_point_name)
    assert attack_point_name in str(verror.value)
