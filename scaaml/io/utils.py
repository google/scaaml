# Copyright 2021-2024 Google LLC
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
"""Utils for scaaml.io"""

from collections import defaultdict
import hashlib
from typing import Callable, DefaultDict, Optional, TypeVar

import tensorflow as tf

T = TypeVar("T")


# TODO: disabling mypy as the ddict() function needs more love for typing
# mypy: ignore-errors
def ddict(value: Optional[DefaultDict[str, Callable[[], T]]], levels: int,
          type_var: Callable[[], T]) -> DefaultDict[str, T]:
    """Returns nested defaultdict of defaultdict (nesting level based on
    level), which are updated with the value dictionary.

    Args:
      value: The dictionary that holds values needs to be either None or the
        same level as the result. All occurrences of None which should be a
        Dict as a value are replaced by an empty defaultdict. Values are
        copied, but not deep-copied.
      levels: Number of defaultdict iterations. Must be non-negative. Level
        zero returns value.
      type_var: The type of the default value.

    Example use:
      >>> d = {"a": 1, "b": 2}
      >>> # e is defaultdict(int) updated by d
      >>> e = ddict(value=d, levels=1, type_var=int)
      >>> e
      defaultdict(<class 'int'>, {"a": 1, "b": 2})
      >>> D = {"A": {"a": 1, "b": 2}, "C": {}}
      >>> # E is defaultdict(lambda: defaultdict(int)) updated by D
      >>> E = ddict(value=D, levels=2, type_var=int)
      >>> E
      defaultdict(<function ddict.<locals>.<lambda> at 0x7fed09f4c790>,
              {"A": defaultdict(<class 'int'>,
                  {"a": 1, "b": 2}),
              "C": defaultdict(<class 'int'>, {})})
      >>> f = ddict(value=None, levels=1, type_var=list)
    """

    def empty_dd(levels: int, type_var: Callable[[], T]) -> DefaultDict[str, T]:
        """Returns the right level of defaultdict."""
        if levels == 1:
            return defaultdict(type_var)
        return defaultdict(lambda: empty_dd(levels - 1, type_var))

    assert levels >= 0
    if levels == 0:
        return value
    result = empty_dd(levels, type_var)
    if value is not None:
        for k in value:
            result[k] = ddict(value[k], levels=levels - 1, type_var=type_var)
    return result


def sha256sum(filename: str) -> str:
    "compute the sha256 of a given file"
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def dtype_name_to_dtype(name: str) -> tf.DType:
    """Turn saved string dtype into tf.dtype.
    """
    if name == "float16":
        return tf.float16
    if name == "float32":
        return tf.float32
    raise ValueError(f"Either float16 or float32 expected, got {name}")


def dtype_dtype_to_name(dtype: tf.DType) -> str:
    """Turn saved string dtype into tf.dtype.
    """
    if dtype == tf.float16:
        return "float16"
    if dtype == tf.float32:
        return "float32"
    raise ValueError(f"Either tf.float16 or tf.float32 expected, got {dtype}")
