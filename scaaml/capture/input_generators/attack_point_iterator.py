# Copyright 2024 Google LLC
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
"""
An Iterator that iterates through attack points 
and can be used with config files.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List


class AttackPointIterator:
    """Attack point iterator class that iterates with different configs."""

    def __init__(self, configuration) -> None:
        """Initialize a new iterator."""
        self._attack_point_iterator_internal: AttackPointIteratorInternalBase
        if configuration["operation"] == "constants":
            self._attack_point_iterator_internal = AttackPointIteratorInternalConstants(
                name=configuration["name"],
                values=configuration["values"],
            )
        else:
            raise ValueError(f"{configuration['operation']} is not supported")

    def __len__(self) -> int:
        """Return the number of iterated elements.
        """
        return len(self._attack_point_iterator_internal)

    def __iter__(self):
        """Start iterating."""
        return iter(self._attack_point_iterator_internal)

    def __next__(self):
        """Next iterated element."""
        return next(self._attack_point_iterator_internal)


class AttackPointIteratorInternalBase(ABC):
    "Attack point iterator abstract class."

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of iterated elements.
        """

    @abstractmethod
    def __iter__(self):
        """Start iterating."""

    @abstractmethod
    def __next__(self):
        """Next iterated element."""


class AttackPointIteratorInternalConstants(AttackPointIteratorInternalBase):
    """Attack point iterator class that iterates over a constant."""

    def __init__(self, name: str, values: List[List[int]]) -> None:
        """Initialize the constants to iterate."""
        self._values_dict = {
            'name': name,
            'value': [],
        }
        self._values = values
        self._index = 0

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.__len__():
            self._values_dict['value'] = self._values[self._index]
            self._index += 1
            return self._values_dict
        else:
            raise StopIteration
