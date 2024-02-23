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

import itertools
import psutil
import numpy as np
import pytest

from scaaml.capture.input_generators import build_attack_points_iterator, LengthIsInfiniteException
from scaaml.capture.input_generators.attack_point_iterator_exceptions import ListNotPrescribedLengthException


def attack_point_iterator_constants(values, length: int = 16):
    config = {
        "operation": "constants",
        "name": "key",
        "length": length,
        "values": values
    }
    output = [
        obj["key"] for obj in list(iter(build_attack_points_iterator(config)))
    ]
    assert output == values
    assert len(values) == len(build_attack_points_iterator(config))


def test_attack_point_iterator_no_legal_operation():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "operation": "NONE",
        "name": "key",
        "length": 16,
        "values": values
    }
    with pytest.raises(ValueError):
        build_attack_points_iterator(config)


def test_attack_point_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    attack_point_iterator_constants(values=values)


def test_single_key_in_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "operation": "constants",
        "name": "key",
        "length": 16,
        "values": values
    }
    for constant in build_attack_points_iterator(config):
        assert list(constant.keys()) == ["key"]


def test_attack_point_iterator_constants_no_values():
    config = {"operation": "constants", "name": "key", "length": 16}
    with pytest.raises(TypeError):
        build_attack_points_iterator(config)


def test_attack_point_iterator_constant_lengths():
    for l in range(4):
        values = np.random.randint(256, size=(l, 17))
        attack_point_iterator_constants(values=values.tolist(), length=17)


def test_attack_point_iterator_constant_length_config():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "operation": "constants",
        "name": "key",
        "length": 16,
        "values": values
    }
    iterator = build_attack_points_iterator(config)
    for values in iterator._values:
        assert len(values) == config["length"]


def test_attack_point_iterator_constant_length_config_missing():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {"operation": "constants", "name": "key", "values": values}
    with pytest.raises(TypeError):
        build_attack_points_iterator(config)


def test_attack_point_iterator_constant_bad_list_length():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "operation": "constants",
        "length": 16,
        "name": "key",
        "values": values
    }
    with pytest.raises(ListNotPrescribedLengthException):
        build_attack_points_iterator(config)


def repeated_iteration(config):
    rep_iterator = build_attack_points_iterator(config)
    assert list(iter(rep_iterator)) == list(iter(rep_iterator))


def test_repeated_iteration_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "operation": "constants",
        "name": "key",
        "length": 16,
        "values": values
    }
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


def test_attack_point_iterator_repeat_default_value():
    config = {
        "operation": "repeat",
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = build_attack_points_iterator(config)
    assert output._repetitions == -1


def test_attack_point_iterator_repeat_zero():
    config = {
        "operation": "repeat",
        "repetitions": 0,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = list(build_attack_points_iterator(config))
    assert output == []


def test_attack_point_iterator_repeat_zero_len():
    config = {
        "operation": "repeat",
        "repetitions": 0,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == 0


def test_attack_point_iterator_repeat_one():
    config = {
        "operation": "repeat",
        "repetitions": 1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert output == list(
        iter(build_attack_points_iterator(config["configuration"])))


def test_attack_point_iterator_repeat_one_len():
    config = {
        "operation": "repeat",
        "repetitions": 1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == len(config["configuration"]["values"])


def test_attack_point_iterator_repeat_two():
    config = {
        "operation": "repeat",
        "repetitions": 2,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = list(iter(build_attack_points_iterator(config)))
    assert output == list(
        iter(build_attack_points_iterator(
            config["configuration"]))) * config["repetitions"]


def test_attack_point_iterator_repeat_two_len():
    config = {
        "operation": "repeat",
        "repetitions": 2,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == config["repetitions"] * len(
        config["configuration"]["values"])


def test_attack_point_iterator_repeat_three_len():
    config = {
        "operation": "repeat",
        "repetitions": 3,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == config["repetitions"] * len(
        config["configuration"]["values"])


def test_attack_point_iterator_repeat_infinite():
    config = {
        "operation": "repeat",
        "repetitions": -1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [[1]]
        }
    }
    count = 0
    for value1, value2 in zip(
            build_attack_points_iterator(config),
            itertools.cycle(
                build_attack_points_iterator(config["configuration"]))):
        if count > 4:
            break
        count += 1
        assert value1 == value2


def test_attack_point_iterator_repeat_infinite_minus_two():
    config = {
        "operation": "repeat",
        "repetitions": -2,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [[1], [2]]
        }
    }
    count = 0
    for value1, value2 in zip(
            build_attack_points_iterator(config),
            itertools.cycle(
                build_attack_points_iterator(config["configuration"]))):
        if count > 4:
            break
        count += 1
        assert value1 == value2


def test_attack_point_iterator_repeat_infinite_no_values():
    config = {
        "operation": "repeat",
        "repetitions": -1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 0,
            "values": []
        }
    }
    output = []
    for count, value in enumerate(build_attack_points_iterator(config)):
        if count > 5:
            break
        output.append(value)

    assert output == []


def test_attack_point_iterator_repeat_infinite_no_values_len():
    config = {
        "operation": "repeat",
        "repetitions": -1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 0,
            "values": []
        }
    }
    output = len(build_attack_points_iterator(config))
    assert output == 0


def test_attack_point_iterator_repeat_infinite_len():
    config = {
        "operation": "repeat",
        "repetitions": -1,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }
    with pytest.raises(LengthIsInfiniteException):
        len(build_attack_points_iterator(config))


def test_repeat_memory():
    config = {
        "operation": "repeat",
        "repetitions": 1_000_000_000_000,
        "configuration": {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [[1], [2], [3]]
        }
    }

    max_growth_factor = 2
    python_process = psutil.Process()
    memory_before_iterator = python_process.memory_info().rss
    memory_threshold = memory_before_iterator * max_growth_factor

    long_iterable = build_attack_points_iterator(config)
    long_iterator = iter(long_iterable)
    assert next(long_iterator) == {"key": [1]}
    assert python_process.memory_info().rss <= memory_threshold
    assert next(long_iterator) == {"key": [2]}
    assert next(long_iterator) == {"key": [3]}
    assert python_process.memory_info().rss <= memory_threshold
    assert next(long_iterator) == {"key": [1]}
    assert next(long_iterator) == {"key": [2]}
    assert next(long_iterator) == {"key": [3]}
    assert next(long_iterator) == {"key": [1]}
    assert python_process.memory_info().rss <= memory_threshold

    assert len(long_iterable) == len(
        config["configuration"]["values"]) * config["repetitions"]
    assert python_process.memory_info().rss <= memory_threshold


def test_attack_point_iterator_zip_same_lengths():
    values = [[0], [1], [2]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values
        }]
    }
    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }, {
        "key": [2],
        "plaintext": [2]
    }]
    assert output_len == len(config["operands"][0]["values"])


def test_attack_point_iterator_zip_different_lengths():
    values = [[0], [1], [2]]
    values2 = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values2
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }]
    assert output_len == len(config["operands"][1]["values"])


def test_attack_point_iterator_zip_different_lengths_length_zero():
    values = []
    values2 = [[0], [1], [2]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values2
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == []
    assert output_len == len(config["operands"][0]["values"])


def test_attack_point_iterator_zip_infinite_and_finite():
    values = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values
            }
        }, {
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values
            }
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }, {
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }]
    assert output_len == len(build_attack_points_iterator(
        config["operands"][1]))


def test_attack_point_iterator_zip_finite_and_infinite():
    values = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values
            }
        }, {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values
            }
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }, {
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }]
    assert output_len == len(build_attack_points_iterator(
        config["operands"][0]))


def test_attack_point_iterator_zip_infinite_and_infinite():
    values = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values
            }
        }, {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values
            }
        }]
    }
    output = build_attack_points_iterator(config)
    count = 0
    for value1, value2 in zip(
            output, zip(itertools.cycle(values), itertools.cycle(values))):
        if count > 4:
            break
        count += 1
        assert value1 == {
            config["operands"][0]["configuration"]["name"]: value2[0],
            config["operands"][1]["configuration"]["name"]: value2[1]
        }
    with pytest.raises(LengthIsInfiniteException):
        len(output)


def test_attack_point_iterator_zip_duplicate_name():
    values = [[0], [1], [2]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }]
    }
    with pytest.raises(ValueError):
        build_attack_points_iterator(config)


def test_attack_point_iterator_zip_three_operands():
    values = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "three",
            "length": 1,
            "values": values
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0],
        "three": [0]
    }, {
        "key": [1],
        "plaintext": [1],
        "three": [1]
    }]
    assert output_len == len(config["operands"][1]["values"])


def test_attack_point_iterator_zip_get_generated_keys():
    values = [[0], [1]]
    config = {
        "operation":
            "zip",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values
        }, {
            "operation": "constants",
            "name": "three",
            "length": 1,
            "values": values
        }]
    }
    output = build_attack_points_iterator(config)

    assert output.get_generated_keys() == [
        config["operands"][0]["name"], config["operands"][1]["name"],
        config["operands"][2]["name"]
    ]


def test_attack_point_iterator_zip_no_operands():
    config = {"operation": "zip", "operands": []}
    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == []
    assert output_len == 0


def test_attack_point_iterator_cartesian_product_two_operands():
    values1 = [[1], [2]]
    values2 = [[4], [5]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values1
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values2
        }]
    }

    output = build_attack_points_iterator(config)
    output_iter = list(iter(output))
    output_len = len(output)

    assert output_iter == [
        {
            "key": [1],
            "plaintext": [4]
        },
        {
            "key": [1],
            "plaintext": [5]
        },
        {
            "key": [2],
            "plaintext": [4]
        },
        {
            "key": [2],
            "plaintext": [5]
        },
    ]
    assert output_len == len(values1) * len(values2)


def test_attack_point_iterator_cartesian_product_three_operands():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    values3 = [[5], [6]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values1
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values2
        }, {
            "operation": "constants",
            "name": "three",
            "length": 1,
            "values": values3
        }]
    }

    output = build_attack_points_iterator(config)

    assert list(iter(output)) == [{
        "key": [1],
        "plaintext": [3],
        "three": [5]
    }, {
        "key": [1],
        "plaintext": [3],
        "three": [6]
    }, {
        "key": [1],
        "plaintext": [4],
        "three": [5]
    }, {
        "key": [1],
        "plaintext": [4],
        "three": [6]
    }, {
        "key": [2],
        "plaintext": [3],
        "three": [5]
    }, {
        "key": [2],
        "plaintext": [3],
        "three": [6]
    }, {
        "key": [2],
        "plaintext": [4],
        "three": [5]
    }, {
        "key": [2],
        "plaintext": [4],
        "three": [6]
    }]
    assert len(output) == 8


def test_attack_point_iterator_cartesian_product_finite_and_no_values():
    values1 = [[1], [2]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values1
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": []
        }]
    }

    output = build_attack_points_iterator(config)

    assert list(iter(output)) == []
    assert len(output) == 0


def test_attack_point_iterator_cartesian_product_infinite_and_no_values():
    values1 = [[1], [2]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values1
            }
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": []
        }]
    }

    output = build_attack_points_iterator(config)

    assert list(iter(output)) == []
    assert len(output) == 0


def test_attack_point_iterator_cartesian_product_finite_and_infinite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values1
        }, {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values2
            }
        }]
    }

    with pytest.raises(LengthIsInfiniteException):
        build_attack_points_iterator(config)


def test_attack_point_iterator_cartesian_product_infinite_and_finite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values1
            }
        }, {
            "operation": "constants",
            "name": "plaintext",
            "length": 1,
            "values": values2
        }]
    }

    with pytest.raises(LengthIsInfiniteException):
        build_attack_points_iterator(config)


def test_attack_point_iterator_cartesian_product_infinite_and_infinite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": values1
            }
        }, {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values2
            }
        }]
    }

    with pytest.raises(LengthIsInfiniteException):
        build_attack_points_iterator(config)


def test_attack_point_iterator_cartesian_product_no_operands():
    config = {"operation": "cartesian_product", "operands": []}
    with pytest.raises(ValueError):
        build_attack_points_iterator(config)


def test_attack_point_iterator_cartesian_product_same_name():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "operation":
            "cartesian_product",
        "operands": [{
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values1
        }, {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": values2
        }]
    }

    with pytest.raises(ValueError):
        build_attack_points_iterator(config)
