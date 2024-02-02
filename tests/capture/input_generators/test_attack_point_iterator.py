# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test attack point iterator."""

import psutil
import numpy as np
import pytest

from scaaml.capture.input_generators import build_attack_points_iterator


def attack_point_iterator_constants(values):
    input = {"operation": "constants", "name": "key", "values": values}
    output = [
        obj["key"] for obj in list(iter(build_attack_points_iterator(input)))
    ]
    assert output == values


def test_attack_point_iterator_no_legal_operation():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    input = {"operation": "NONE", "name": "key", "values": values}
    with pytest.raises(ValueError):
        build_attack_points_iterator(input)


def test_attack_point_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    attack_point_iterator_constants(values=values)


def test_single_key_in_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    input = {"operation": "constants", "name": "key", "values": values}
    for constant in build_attack_points_iterator(input):
        assert list(constant.keys()) == ["key"]


def test_attack_point_iterator_constants_no_values():
    input = {"operation": "constants", "name": "key"}
    with pytest.raises(TypeError):
        build_attack_points_iterator(input)


def test_attack_point_iterator_constant_lengths():
    for l in range(4):
        values = np.random.randint(256, size=(l, 17))
        attack_point_iterator_constants(values=values.tolist())


def repeated_iteration(config):
    rep_iterator = build_attack_points_iterator(config)
    assert list(iter(rep_iterator)) == list(iter(rep_iterator))


def test_repeated_iteration_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {"operation": "constants", "name": "key", "values": values}
    repeated_iteration(config)


def test_attack_point_iterator_balanced_generator():
    config = {"operation": "balanced_generator", "name": "key", "length": 16}
    output = list(iter(build_attack_points_iterator(config)))
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_kwargs():
    config = {
        "operation": "balanced_generator",
        "name": "key",
        "length": 16,
        "bunches": 2,
        "elements": 3
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert isinstance(config["bunches"], int)
    assert isinstance(config["elements"], int)
    assert len(output) == config["bunches"] * config["elements"]


def test_attack_point_iterator_unrestricted_generator():
    config = {
        "operation": "unrestricted_generator",
        "name": "key",
        "length": 16
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_args():
    config = {
        "operation": "unrestricted_generator",
        "name": "key",
        "length": 16,
        "bunches": 2,
        "elements": 3
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert len(output) == config["bunches"] * config["elements"]


def test_attack_point_iterator_balanced_generator_len():
    config = {"operation": "balanced_generator", "name": "key", "length": 16}
    output = build_attack_points_iterator(config)
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_args_len():
    config = {
        "operation": "balanced_generator",
        "name": "key",
        "length": 16,
        "bunches": 2,
        "elements": 3
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert len(output) == config["bunches"] * config["elements"]
    assert len(output) == len(build_attack_points_iterator(config))


def test_attack_point_iterator_unrestricted_generator_all_args_len():
    config = {
        "operation": "unrestricted_generator",
        "name": "key",
        "length": 16,
        "bunches": 2,
        "elements": 3
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert len(output) == config["bunches"] * config["elements"]
    assert len(output) == len(build_attack_points_iterator(config))


def test_attack_point_iterator_repeat():
    config = {
        "operation": "repeat",
        "repetitions": 2,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "values": [1, 2, 3]
        }
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert output == list(
        iter(build_attack_points_iterator(
            config["configuration"]))) * config["repetitions"]

def test_attack_point_iterator_repeat_infinite_len():
    config = {
        "operation": "repeat",
        "repetitions": 0,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "values": [1, 2, 3]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == 2**128 * len(
        config["configuration"]["values"])

def test_attack_point_iterator_repeat_len():
    config = {
        "operation": "repeat",
        "repetitions": 2,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "values": [1, 2, 3]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == config["repetitions"] * len(
        config["configuration"]["values"])


def test_repeat_memory():
    config = {
        "operation": "repeat",
        "repetitions": 1_000_000_000_000,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "values": [[1], [2], [3]]
        }
    }

    max_growth_factor = 2
    python_process = psutil.Process()
    memory_before_iterator = python_process.memory_info().rss

    long_iterable = build_attack_points_iterator(config)
    long_iterator = iter(long_iterable)
    assert next(long_iterator) == {"key": [1]}
    assert python_process.memory_info(
    ).rss <= max_growth_factor * memory_before_iterator
    assert next(long_iterator) == {"key": [2]}
    assert next(long_iterator) == {"key": [3]}
    assert python_process.memory_info(
    ).rss <= max_growth_factor * memory_before_iterator
    assert next(long_iterator) == {"key": [1]}
    assert next(long_iterator) == {"key": [2]}
    assert next(long_iterator) == {"key": [3]}
    assert next(long_iterator) == {"key": [1]}
    assert python_process.memory_info(
    ).rss <= max_growth_factor * memory_before_iterator

    assert len(long_iterable) == len(
        config["configuration"]["values"]) * config["repetitions"]
    assert python_process.memory_info(
    ).rss <= max_growth_factor * memory_before_iterator
