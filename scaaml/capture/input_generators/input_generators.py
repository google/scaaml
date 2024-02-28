# Copyright 2023-2024 Google LLC
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
"""Generating random attack point values. Can be also used for masks etc.
Usually represented as arrays of byte values.
"""

from typing import Iterator, List, Optional

import numpy as np
import numpy.typing as npt

InputGeneratorT = npt.NDArray[np.int64]


def single_bunch(length: int,
                 elements: int = 256,
                 seed: Optional[int] = None) -> InputGeneratorT:
    """Generate a single bunch of arrays (2D array -- rows * columns). Each row
    has elements chosen uniformly independently from range(elements). But two
    rows have no index where they are equal.

    Rationale: neural networks are very good at picking statistical properties.
    By random chance the value 0x2A of byte 3 (concrete values chosen for
    illustration purposes only) appears more often than others and this is
    consistent in train and valuation set. Then we can get false positives (it
    looks like the neural network is sort of working). Generating the train set
    such that each byte attains each value the same number of times helps to
    mitigate this situation.

    Explanation of the algorithm: for each index we create a random permutation
    of elements and concatenate those, we then return this transposed.

    Beware of doing two for loops over two single_bunch outputs generated with
    the same seed. This would return identical arrays.

    Args:
      length (int): Length of each of the resulting arrays.
      elements (int): Each array contains numbers in `range(elements)`.
      seed (Optional[int]): Seed for the random generator. Passed to
        numpy.random.default_rng. For testing purposes only.

    Returns: an np.array of shape `(elements, length)` where for each index
      `i in range(length)` and for each `value in range(elements)` there is an
      array which contains `value` on position `i`.

    Example use:
      >>> print("Small and deterministic example:")
      >>> single_bunch(5, 3, 42)
      array([[2, 0, 1, 2, 2],
             [1, 2, 2, 0, 0],
             [0, 1, 0, 1, 1]])
      >>> print("Realistic example:")
      >>> for value in single_bunch(32):
      >>>     print(f{value} has each byte chosen uniformly at random)
      >>> print("But no two values had same element on the same position.")
    """
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    transposed_result: List[InputGeneratorT] = []

    for _ in range(length):
        # One column (content of result[:, i]).
        column: InputGeneratorT = np.arange(elements, dtype=np.int64)

        # permuted does a copy, just to be sure rows are not constant.
        transposed_result.append(rng.permuted(column))

    # Convert to np
    result: InputGeneratorT = np.array(transposed_result, dtype=np.int64)

    # Transpose the output to get a list of arrays
    result = np.transpose(result)

    return result


def balanced_generator(length: int,
                       bunches: int = 1,
                       elements: int = 256) -> Iterator[InputGeneratorT]:
    """Generator of values for the training set. Feel free to use
    unrestricted_generator for holdout, but not for training split.

    Args:
      length (int): Length of each yielded array.
      bunches (int): How many bunches to return (yields `bunches * elements`
        random arrays).
      elements (int): Each array contains numbers in `range(elements)`.

    Example use: Output may vary based on randomly generated numbers.

    Two bunches examples (generates bunches * elements arrays):
    >>> for value in balanced_generator(5, bunches=2, elements=3):
    >>>     print(value)
    [0 1 2 2 1]
    [1 2 1 0 2]
    [2 0 0 1 0]
    [1 1 0 1 1]
    [2 0 2 2 2]
    [0 2 1 0 0]
    >>> print("Each value repeats the same number of times in each column.")
    >>> print("But columns may repeat.")

    AES128 example: for each key have balanced plain_texts (thus SBOX inputs are
    also balanced). Note that length is number of bytes = 16 (128 bits / 8).
    This results in 65,536 traces, 256 different keys and for each key 256
    different plain_texts (for different keys we generate different plain_texts
    at random).
    >>> for key in balanced_generator(length=16):
    >>>     for plain_text in balanced_generator(length=16):
    >>>         print(f"Capture trace with {key = } and {plain_text = }")

    AES128-like example (small, output may be different, based on random
    chance):
    >>> for key in balanced_generator(length=5, elements=3):
    >>>     for plain_text in balanced_generator(length=5, elements=3):
    >>>         print(f"Capture trace with {key = } and {plain_text = }")
    key = array([0, 2, 0, 0, 0]) plain_text = array([0, 1, 1, 1, 0])
    key = array([0, 2, 0, 0, 0]) plain_text = array([2, 2, 0, 2, 2])
    key = array([0, 2, 0, 0, 0]) plain_text = array([1, 0, 2, 0, 1])
    key = array([2, 1, 1, 2, 2]) plain_text = array([2, 0, 0, 2, 2])
    key = array([2, 1, 1, 2, 2]) plain_text = array([1, 1, 2, 0, 1])
    key = array([2, 1, 1, 2, 2]) plain_text = array([0, 2, 1, 1, 0])
    key = array([1, 0, 2, 1, 1]) plain_text = array([1, 0, 2, 0, 0])
    key = array([1, 0, 2, 1, 1]) plain_text = array([2, 2, 1, 1, 2])
    key = array([1, 0, 2, 1, 1]) plain_text = array([0, 1, 0, 2, 1])

    Get numpy array of all results:
    >>> np.array(list(balanced_generator(5, bunches=2, elements=3)))
    """
    for _ in range(bunches):
        bunch: InputGeneratorT = single_bunch(
            length=length,
            elements=elements,
            seed=None,  # random
        )

        # yield each row
        yield from bunch


def unrestricted_generator(length: int,
                           bunches: int = 1,
                           elements: int = 256) -> Iterator[InputGeneratorT]:
    """Do not use this for the training set, use balanced_generator instead.
    Each element is chosen uniformly at random independently from others.

    Args:
      length (int): Length of each yielded array.
      bunches (int): How many bunches to return (yields `bunches * elements`
        random arrays).
      elements (int): Each array contains numbers in `range(elements)`.

    Example use: there is a single bunch, but the first two arrays have the
    same value of the last element. Check balanced_generator where this cannot
    happen. Output may vary based on randomly generated numbers.
    >>> for value in unrestricted_generator(5, elements=3):
    >>>     print(value)
    [0 1 1 1 1]
    [1 0 2 0 1]
    [2 1 1 2 0]
    """
    rng: np.random.Generator = np.random.default_rng(seed=None)

    for _ in range(bunches * elements):
        yield rng.integers(low=0, high=elements, size=length, dtype=np.int64)
