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
from pydantic import ValidationError
import pytest

from scaaml.capture.input_generators import IteratorModel, LengthIsInfiniteException, ListNotPrescribedLengthException


def attack_point_iterator_constants(values, length: int = 16):
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": length,
            "values": values
        }
    }
    iterator = IteratorModel.model_validate(config)
    output = [obj["key"] for obj in list(iter(iterator.items()))]
    assert output == values
    assert len(values) == len(iterator)


def test_attack_point_iterator_no_legal_operation():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "NONE",
            "name": "key",
            "length": 16,
            "values": values
        }
    }
    with pytest.raises(ValueError):
        IteratorModel.model_validate(config)


def test_attack_point_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    attack_point_iterator_constants(values=values)


def test_single_key_in_iterator_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 16,
            "values": values
        }
    }
    for constant in IteratorModel.model_validate(config).items():
        assert list(constant.keys()) == ["key"]


def test_attack_point_iterator_constants_no_values():
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 16
        }
    }
    with pytest.raises(ValidationError):
        IteratorModel.model_validate(config)


def test_attack_point_iterator_constant_lengths():
    for l in range(4):
        values = np.random.randint(256, size=(l, 17))
        attack_point_iterator_constants(values=values.tolist(), length=17)


def test_attack_point_iterator_constant_length_config():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 16,
            "values": values
        }
    }
    iterator = IteratorModel.model_validate(config)
    for values in iterator.iterator_model.values:
        assert len(values) == config["iterator_model"]["length"]


def test_attack_point_iterator_constant_length_config_missing():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "values": values
        }
    }
    with pytest.raises(ValidationError):
        IteratorModel.model_validate(config)


def test_attack_point_iterator_constant_bad_list_length():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "constants",
            "length": 16,
            "name": "key",
            "values": values
        }
    }
    with pytest.raises(ListNotPrescribedLengthException):
        IteratorModel.model_validate(config)


def repeated_iteration(config):
    rep_iterator = IteratorModel.model_validate(config)
    assert list(iter(rep_iterator)) == list(iter(rep_iterator))


def test_repeated_iteration_constants():
    values = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    config = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 16,
            "values": values
        }
    }
    repeated_iteration(config)


def test_attack_point_iterator_balanced_generator():
    config = {
        "iterator_model": {
            "operation": "balanced_generator",
            "name": "key",
            "length": 16
        }
    }
    output = list(iter(IteratorModel.model_validate(config).items()))
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_kwargs():
    config = {
        "iterator_model": {
            "operation": "balanced_generator",
            "name": "key",
            "length": 16,
            "bunches": 2,
            "elements": 3
        }
    }
    output = list(iter(IteratorModel.model_validate(config).items()))
    assert isinstance(config["iterator_model"]["bunches"], int)
    assert isinstance(config["iterator_model"]["elements"], int)
    assert len(output) == config["iterator_model"]["bunches"] * config[
        "iterator_model"]["elements"]


def test_attack_point_iterator_unrestricted_generator():
    config = {
        "iterator_model": {
            "operation": "unrestricted_generator",
            "name": "key",
            "length": 16
        }
    }
    output = list(iter(IteratorModel.model_validate(config).items()))
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_args():
    config = {
        "iterator_model": {
            "operation": "unrestricted_generator",
            "name": "key",
            "length": 16,
            "bunches": 2,
            "elements": 3
        }
    }
    output = list(iter(IteratorModel.model_validate(config).items()))
    assert len(output) == config["iterator_model"]["bunches"] * config[
        "iterator_model"]["elements"]


def test_attack_point_iterator_balanced_generator_len():
    config = {
        "iterator_model": {
            "operation": "balanced_generator",
            "name": "key",
            "length": 16
        }
    }
    output = IteratorModel.model_validate(config)
    assert len(output) == 256


def test_attack_point_iterator_balanced_generator_all_args_len():
    config = {
        "iterator_model": {
            "operation": "balanced_generator",
            "name": "key",
            "length": 16,
            "bunches": 2,
            "elements": 3
        }
    }
    iterator = IteratorModel.model_validate(config)
    output = list(iter(iterator.items()))
    assert len(output) == config["iterator_model"]["bunches"] * config[
        "iterator_model"]["elements"]
    assert len(output) == len(iterator)


def test_attack_point_iterator_unrestricted_generator_all_args_len():
    config = {
        "iterator_model": {
            "operation": "unrestricted_generator",
            "name": "key",
            "length": 16,
            "bunches": 2,
            "elements": 3
        }
    }
    iterator = IteratorModel.model_validate(config)
    output = list(iter(iterator.items()))
    assert len(output) == config["iterator_model"]["bunches"] * config[
        "iterator_model"]["elements"]
    assert len(output) == len(iterator)


def test_attack_point_iterator_repeat_default_value():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    output = IteratorModel.model_validate(config)
    assert output.iterator_model.repetitions == -1


def test_attack_point_iterator_repeat_zero():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 0,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    output = list(IteratorModel.model_validate(config).items())
    assert output == []


def test_attack_point_iterator_repeat_zero_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 0,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    output = len(IteratorModel.model_validate(config))
    assert output == 0


def test_attack_point_iterator_repeat_one():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }

    configuration = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }

    output = list(iter(IteratorModel.model_validate(config).items()))
    assert output == list(
        iter(IteratorModel.model_validate(configuration).items()))


def test_attack_point_iterator_repeat_one_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }

    output = len(IteratorModel.model_validate(config))
    assert output == len(config["iterator_model"]["configuration"]["values"])


def test_attack_point_iterator_repeat_two():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }

    configuration = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 3,
            "values": [[1, 2, 3]]
        }
    }

    output = list(iter(IteratorModel.model_validate(config).items()))
    assert output == list(
        iter(IteratorModel.model_validate(
            configuration).items())) * config["iterator_model"]["repetitions"]


def test_attack_point_iterator_repeat_two_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    output = len(IteratorModel.model_validate(config))
    assert output == config["iterator_model"]["repetitions"] * len(
        config["iterator_model"]["configuration"]["values"])


def test_attack_point_iterator_repeat_three_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 3,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    output = len(IteratorModel.model_validate(config))
    assert output == config["iterator_model"]["repetitions"] * len(
        config["iterator_model"]["configuration"]["values"])


def test_attack_point_iterator_repeat_infinite():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": [[1]]
            }
        }
    }
    configuration = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [[1]]
        }
    }
    count = 0
    for value1, value2 in zip(
            IteratorModel.model_validate(config).items(),
            itertools.cycle(
                IteratorModel.model_validate(configuration).items())):
        if count > 4:
            break
        count += 1
        assert value1 == value2


def test_attack_point_iterator_repeat_infinite_minus_two():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": -2,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": [[1], [2]]
            }
        }
    }

    configuration = {
        "iterator_model": {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [[1], [2]]
        }
    }
    count = 0
    for value1, value2 in zip(
            IteratorModel.model_validate(config).items(),
            itertools.cycle(
                IteratorModel.model_validate(configuration).items())):
        if count > 4:
            break
        count += 1
        assert value1 == value2


def test_attack_point_iterator_repeat_infinite_no_values():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 0,
                "values": []
            }
        }
    }
    output = []
    for count, value in enumerate(IteratorModel.model_validate(config).items()):
        if count > 5:
            break
        output.append(value)

    assert output == []


def test_attack_point_iterator_repeat_infinite_no_values_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 0,
                "values": []
            }
        }
    }
    output = len(IteratorModel.model_validate(config))
    assert output == 0


def test_attack_point_iterator_repeat_infinite_len():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": -1,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 3,
                "values": [[1, 2, 3]]
            }
        }
    }
    with pytest.raises(LengthIsInfiniteException):
        len(IteratorModel.model_validate(config))


def test_repeat_memory():
    config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 1_000_000_000_000,
            "configuration": {
                "operation": "constants",
                "name": "key",
                "length": 1,
                "values": [[1], [2], [3]]
            }
        }
    }

    max_growth_factor = 2
    python_process = psutil.Process()
    memory_before_iterator = python_process.memory_info().rss
    memory_threshold = memory_before_iterator * max_growth_factor

    long_iterable = IteratorModel.model_validate(config)
    long_iterator = iter(long_iterable.items())
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
        config["iterator_model"]["configuration"]
        ["values"]) * config["iterator_model"]["repetitions"]
    assert python_process.memory_info().rss <= memory_threshold


def test_attack_point_iterator_zip_same_lengths():
    values = [[0], [1], [2]]
    config = {
        "iterator_model": {
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
    }
    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
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
    assert output_len == len(config["iterator_model"]["operands"][0]["values"])


def test_attack_point_iterator_zip_different_lengths():
    values = [[0], [1], [2]]
    values2 = [[0], [1]]
    config = {
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
    output_len = len(output)

    assert output_iter == [{
        "key": [0],
        "plaintext": [0]
    }, {
        "key": [1],
        "plaintext": [1]
    }]
    assert output_len == len(config["iterator_model"]["operands"][1]["values"])


def test_attack_point_iterator_zip_different_lengths_length_zero():
    values = []
    values2 = [[0], [1], [2]]
    config = {
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
    output_len = len(output)

    assert output_iter == []
    assert output_len == len(config["iterator_model"]["operands"][0]["values"])


def test_attack_point_iterator_zip_infinite_and_finite():
    values = [[0], [1]]
    config = {
        "iterator_model": {
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
    }
    length_config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values
            }
        }
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
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
    assert output_len == len(IteratorModel.model_validate(length_config))


def test_attack_point_iterator_zip_finite_and_infinite():
    values = [[0], [1]]
    config = {
        "iterator_model": {
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
    }

    len_config = {
        "iterator_model": {
            "operation": "repeat",
            "repetitions": 2,
            "configuration": {
                "operation": "constants",
                "name": "plaintext",
                "length": 1,
                "values": values
            }
        }
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
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
    assert output_len == len(IteratorModel.model_validate(len_config))


def test_attack_point_iterator_zip_infinite_and_infinite():
    values = [[0], [1]]
    config = {
        "iterator_model": {
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
    }
    output = IteratorModel.model_validate(config)
    count = 0
    for value1, value2 in zip(
            output.items(), zip(itertools.cycle(values),
                                itertools.cycle(values))):
        if count > 4:
            break
        count += 1
        assert value1 == {
            config["iterator_model"]["operands"][0]["configuration"]["name"]:
                value2[0],
            config["iterator_model"]["operands"][1]["configuration"]["name"]:
                value2[1]
        }
    with pytest.raises(LengthIsInfiniteException):
        len(output)


def test_attack_point_iterator_zip_duplicate_name():
    values = [[0], [1], [2]]
    config = {
        "iterator_model": {
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
    }
    with pytest.raises(ValueError):
        IteratorModel.model_validate(config)


def test_attack_point_iterator_zip_three_operands():
    values = [[0], [1]]
    config = {
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
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
    assert output_len == len(config["iterator_model"]["operands"][1]["values"])


def test_attack_point_iterator_zip_get_generated_keys():
    values = [[0], [1]]
    config = {
        "iterator_model": {
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
    }
    output = IteratorModel.model_validate(config)

    assert output.iterator_model.get_generated_keys() == [
        config["iterator_model"]["operands"][0]["name"],
        config["iterator_model"]["operands"][1]["name"],
        config["iterator_model"]["operands"][2]["name"]
    ]


def test_attack_point_iterator_zip_no_operands():
    config = {"iterator_model": {"operation": "zip", "operands": []}}
    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
    output_len = len(output)

    assert output_iter == []
    assert output_len == 0


def test_attack_point_iterator_cartesian_product_two_operands():
    values1 = [[1], [2]]
    values2 = [[4], [5]]
    config = {
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)
    output_iter = list(iter(output.items()))
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
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)

    assert list(iter(output.items())) == [{
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
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)

    assert list(iter(output.items())) == []
    assert len(output) == 0


def test_attack_point_iterator_cartesian_product_infinite_and_no_values():
    values1 = [[1], [2]]
    config = {
        "iterator_model": {
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
    }

    output = IteratorModel.model_validate(config)

    assert list(iter(output.items())) == []
    assert len(output) == 0


def test_attack_point_iterator_cartesian_product_finite_and_infinite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "iterator_model": {
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
    }

    with pytest.raises(LengthIsInfiniteException):
        len(IteratorModel.model_validate(config))


def test_attack_point_iterator_cartesian_product_infinite_and_finite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "iterator_model": {
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
    }

    with pytest.raises(LengthIsInfiniteException):
        len(IteratorModel.model_validate(config))


def test_attack_point_iterator_cartesian_product_infinite_and_infinite():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "iterator_model": {
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
    }
    with pytest.raises(LengthIsInfiniteException):
        len(IteratorModel.model_validate(config))


def test_attack_point_iterator_cartesian_product_no_operands():
    config = {"operation": "cartesian_product", "operands": []}
    with pytest.raises(ValueError):
        IteratorModel.model_validate(config)


def test_attack_point_iterator_cartesian_product_same_name():
    values1 = [[1], [2]]
    values2 = [[3], [4]]
    config = {
        "iterator_model": {
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
    }

    with pytest.raises(ValueError):
        IteratorModel.model_validate(config)
