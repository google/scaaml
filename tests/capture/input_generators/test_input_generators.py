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
"""Test attack point generation.
"""

import numpy as np

from scaaml.stats import APChecker, APCounter
from scaaml.capture.input_generators import balanced_generator
from scaaml.capture.input_generators import single_bunch
from scaaml.capture.input_generators import unrestricted_generator


def test_single_bunch_doc():
    """Test docstring."""
    answer = single_bunch(length=5, elements=3, seed=42)

    expected = np.array([[2, 0, 1, 2, 2], [1, 2, 2, 0, 0], [0, 1, 0, 1, 1]],
                        dtype=np.int64)

    assert np.array_equal(answer, expected)


def test_single_bunch_shape():
    """Test shape."""
    for length in range(1, 25):
        for elements in range(1, 25):
            answer = single_bunch(length=length, elements=elements, seed=42)

            assert answer.shape == (elements, length)


def test_single_bunch_counts():
    """Test counts."""
    for length in range(1, 25):
        for elements in range(1, 25):
            answer = single_bunch(length=length, elements=elements)

            for row_2 in range(elements):
                for row_1 in range(row_2):
                    for i in range(length):
                        assert answer[row_1][i] != answer[row_2][i]


def test_balanced_generator_no_repeat():
    """Test that no two attack points are equal. By chance they could be equal,
    but the chance is negligible."""
    length = 16
    bunches = 3
    elements = 256

    all_values = set(
        str(value) for value in balanced_generator(
            length=length, bunches=bunches, elements=elements))

    assert len(all_values) == bunches * elements


def test_balanced_generator_two_no_repeat_full():
    """Test AES128 use-case."""
    length = 16
    bunches = 1  # keep low
    elements = 256
    all_values = set()

    for key in balanced_generator(length=length,
                                  bunches=bunches,
                                  elements=elements):
        for plain_text in balanced_generator(length=length,
                                             bunches=bunches,
                                             elements=elements):
            all_values.add(f"{str(key)} -  {str(plain_text)}")

    assert len(all_values) == bunches * bunches * elements * elements


def test_balanced_generator_two_no_repeat_bunches():
    """Test AES128-like use-case."""
    length = 61
    bunches = 3  # keep low
    elements = 4
    all_values = set()

    for key in balanced_generator(length=length,
                                  bunches=bunches,
                                  elements=elements):
        for plain_text in balanced_generator(length=length,
                                             bunches=bunches,
                                             elements=elements):
            all_values.add(f"{str(key)} -  {str(plain_text)}")

    assert len(all_values) == bunches * bunches * elements * elements


def test_unrestricted_generator_two_no_repeat():
    """Test AES128-like use-case."""
    length = 160
    bunches = 3  # keep low
    elements = 7
    all_values = set()

    for key in unrestricted_generator(length=length,
                                      bunches=bunches,
                                      elements=elements):
        for plain_text in unrestricted_generator(length=length,
                                                 bunches=bunches,
                                                 elements=elements):
            all_values.add(f"{str(key)} -  {str(plain_text)}")

    assert len(all_values) == bunches * bunches * elements * elements


def test_balanced_generator_two_different_attack_point_values():
    """Test AES128-like use-case."""
    length = 128
    bunches = 3  # keep low
    elements = 6
    all_attack_point_1_values = set()
    all_attack_point_2_values = set()

    for attack_point_1 in balanced_generator(length=length,
                                             bunches=bunches,
                                             elements=elements):
        for attack_point_2 in balanced_generator(length=length,
                                                 bunches=bunches,
                                                 elements=elements):
            all_attack_point_1_values.add(str(attack_point_1))
            all_attack_point_2_values.add(str(attack_point_2))

    assert len(all_attack_point_1_values) == bunches * elements

    # Different values for each attack_point_1 value
    assert len(
        all_attack_point_2_values) == bunches * bunches * elements * elements

    assert all_attack_point_1_values.isdisjoint(all_attack_point_2_values)


def test_unrestricted_generator_two_different_attack_point_values():
    """Test AES128-like use-case."""
    length = 128
    bunches = 3  # keep low
    elements = 8
    all_attack_point_1_values = set()
    all_attack_point_2_values = set()

    for attack_point_1 in unrestricted_generator(length=length,
                                                 bunches=bunches,
                                                 elements=elements):
        for attack_point_2 in unrestricted_generator(length=length,
                                                     bunches=bunches,
                                                     elements=elements):
            all_attack_point_1_values.add(str(attack_point_1))
            all_attack_point_2_values.add(str(attack_point_2))

    assert len(all_attack_point_1_values) == bunches * elements

    # Different values for each attack_point_1 value
    assert len(
        all_attack_point_2_values) == bunches * bunches * elements * elements

    assert all_attack_point_1_values.isdisjoint(all_attack_point_2_values)


def test_using_statistical_tests():
    """Test using scaaml.stats"""
    length = 17
    bunches = 8
    elements = 123
    ap_counter = APCounter({"len": length, "max_val": elements})

    for value in balanced_generator(length=length,
                                    bunches=bunches,
                                    elements=elements):
        ap_counter.update(value)

    checker = APChecker(counts=ap_counter.get_counts(),
                        attack_point_name="value")
    checker.run_all()
    assert not checker._something_failed
