"""Test attack point iterator."""

import pytest

from scaaml.capture.input_generators import AttackPointIterator


def test_attack_point_itarattor_no_legal_operation():
    input = {
        "operation": "NONE",
        "name": "key",
        "values": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                   [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    }
    with pytest.raises(ValueError):
        AttackPointIterator(input)


def test_attack_point_iterator_constants():
    input = {
        "operation": "constants",
        "name": "key",
        "values": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                   [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    }
    output = []
    for constant in AttackPointIterator(input):
        output.append(constant.value)
    assert output == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]


def test_attack_point_iterator_constants_no_values():
    input = {
        "operation": "constants",
        "name": "key"
    }
    output = []
    with pytest.raises(KeyError):
        AttackPointIterator(input)
