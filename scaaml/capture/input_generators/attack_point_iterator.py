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
import collections
import copy
import itertools
from typing import Dict, List

from scaaml.capture.input_generators.input_generators import balanced_generator, unrestricted_generator


class AttackPointIterator(ABC):
    "Attack point iterator abstract class."

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of iterated elements."""

    @abstractmethod
    def __iter__(self):
        """Start iterating."""

    @abstractmethod
    def get_generated_keys(self) -> List[str]:
        """
        Returns an exhaustive list of names this iterator 
        and its children will create.
        """


class LengthIsInfiniteException(Exception):
    """This exception get's called when a number is infinite
      and wants to be represented by __len__ function."""


def build_attack_points_iterator(configuration: Dict) -> AttackPointIterator:
    configuration = copy.deepcopy(configuration)
    iterator = _build_attack_points_iterator(configuration)

    # Check that all names are unique
    names_list = collections.Counter(iterator.get_generated_keys())
    duplicates = [name for name, count in names_list.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated attack point names {duplicates}")

    return iterator


def _build_attack_points_iterator(configuration: Dict) -> AttackPointIterator:
    supported_operations = {
        "constants": AttackPointIteratorConstants,
        "balanced_generator": AttackPointIteratorBalancedGenerator,
        "unrestricted_generator": AttackPointIteratorUnrestrictedGenerator,
        "repeat": AttackPointIteratorRepeat,
    }
    operation = configuration["operation"]
    iterator_cls = supported_operations.get(operation)

    if iterator_cls is None:
        raise ValueError(f"Operation {operation} not supported")

    return iterator_cls(**configuration)


class AttackPointIteratorConstants(AttackPointIterator):
    """Attack point iterator class that iterates over a constant."""

    def __init__(self, operation: str, name: str,
                 values: List[List[int]]) -> None:
        """Initialize the constants to iterate."""
        assert "constants" == operation
        self._name = name
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter({self._name: value} for value in self._values)

    def get_generated_keys(self) -> List[str]:
        return [self._name]


class AttackPointIteratorBalancedGenerator(AttackPointIterator):
    """
    Attack point iterator class that iterates over the balanced generator.
    """

    def __init__(self,
                 operation: str,
                 name: str,
                 length: int,
                 bunches: int = 1,
                 elements: int = 256) -> None:
        """Initialize the balanced kwargs to iterate."""
        assert "balanced_generator" == operation
        self._name = name
        self._length = length
        self._bunches = bunches
        self._elements = elements
        self._len = self._bunches * self._elements

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return iter({self._name: value}
                    for value in balanced_generator(length=self._length,
                                                    bunches=self._bunches,
                                                    elements=self._elements))

    def get_generated_keys(self) -> List[str]:
        return [self._name]


class AttackPointIteratorUnrestrictedGenerator(AttackPointIterator):
    """
    This exception is raised when the `__len__` function is called
    on an infinite iterator.
    """

    def __init__(self,
                 operation: str,
                 name: str,
                 length: int,
                 elements: int = 256,
                 bunches: int = 1) -> None:
        """Initialize the unrestricted kwargs to iterate."""
        assert "unrestricted_generator" == operation
        self._name = name
        self._length = length
        self._elements = elements
        self._bunches = bunches
        self._len = self._bunches * self._elements

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return iter({self._name: value} for value in unrestricted_generator(
            length=self._length, bunches=self._bunches,
            elements=self._elements))

    def get_generated_keys(self) -> List[str]:
        return [self._name]


class AttackPointIteratorRepeat(AttackPointIterator):
    """
    Attack point iterator class that iterates 
    over a configuration a repeat amount of times.
    """

    def __init__(self,
                 operation: str,
                 configuration: Dict,
                 repetitions: int = -1) -> None:
        """Initialize the repeated iterate if repetitions is not present
          or set to a negative number it will do an infinite loop 
          if it is 0 it will not repeat at all.
          
          Args:
            operation (str): The operation of the iterator
                this gets asserted at the start.
            configuration (Dict): The config for the iterated object
                that will get repeated.
            repetitions (int): This parameter decides how often the
                iterator gets repeated. If it is a negative number it
                will iterate infinitely. If it is 0 then it will not
                iterate at all. If it is a positive number it will
                iterate that many times.
            """
        assert "repeat" == operation
        self._configuration_iterator = build_attack_points_iterator(
            configuration)
        if repetitions >= 0:
            self._repetitions = repetitions
            self._len = repetitions * len(self._configuration_iterator)
        elif len(self._configuration_iterator) == 0:
            self._repetitions = 0
            self._len = 0
        else:
            self._repetitions = repetitions
            self._len = repetitions

    def __len__(self) -> int:
        if self._len < 0:
            raise LengthIsInfiniteException("The length is infinite!")
        return self._len

    def __iter__(self):
        if self._repetitions < 0:
            return iter(itertools.cycle(self._configuration_iterator))
        return iter(value for _ in range(self._repetitions)
                    for value in self._configuration_iterator)

    def get_generated_keys(self) -> List[str]:
        return self._configuration_iterator.get_generated_keys()
