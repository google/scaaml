"""Test attack point iterator."""

import numpy as np
import pytest

from scaaml.capture.input_generators import AttackPointIterator


def attack_point_iterator_constants(values):
    input = {"operation": "constants", "name": "key", "values": values}
    output = [obj['key'] for obj in list(iter(AttackPointIterator(input)))]
    assert output == values


def test_attack_point_itarattor_no_legal_operation():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    input = {"operation": "NONE", "name": "key", "values": values}
    with pytest.raises(ValueError):
        AttackPointIterator(input)


def test_attack_point_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    attack_point_iterator_constants(values=values)


def test_single_key_in_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    input = {"operation": "constants", "name": "key", "values": values}
    for constant in AttackPointIterator(input):
        assert list(constant.keys()) == ["key"]


def test_attack_point_iterator_constants_no_values():
    input = {"operation": "constants", "name": "key"}
    output = []
    with pytest.raises(KeyError):
        AttackPointIterator(input)


def test_attack_point_iterator_constant_lengths():
    for l in range(4):
        values = np.random.randint(256, size=(l, 17))
        attack_point_iterator_constants(values=values.tolist())


def repeated_iteration(config):
    rep_iterator = AttackPointIterator(config)
    assert list(iter(rep_iterator)) == list(iter(rep_iterator))


def test_repeated_iteration_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {"operation": "constants", "name": "key", "values": values}
    repeated_iteration(config)
