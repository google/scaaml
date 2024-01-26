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
from typing import Dict, List

from scaaml.capture.input_generators.input_generators import balanced_generator, unrestricted_generator


class AttackPointIterator:
    """Attack point iterator class that iterates with different configs."""

    def __init__(self, configuration) -> None:
        """Initialize a new iterator."""
        self._attack_point_iterator_internal: AttackPointIteratorInternalBase
        if configuration["operation"] == "constants":
            constant_iter = AttackPointIteratorInternalConstants(
                name=configuration["name"],
                values=configuration["values"],
            )
            self._attack_point_iterator_internal = constant_iter
        elif configuration["operation"] == "balanced_generator":
            balanced_iter = AttackPointIteratorInternalBalancedGenerator(
                name=configuration["name"], kwargs=configuration["kwargs"])
            self._attack_point_iterator_internal = balanced_iter
        elif configuration["operation"] == "unrestricted_generator":
            unrestricted = AttackPointIteratorInternalUnrestrictedGenerator(
                name=configuration["name"], kwargs=configuration["kwargs"])
            self._attack_point_iterator_internal = unrestricted
        else:
            raise ValueError(f"{configuration['operation']} is not supported")

    def __len__(self) -> int:
        """Return the number of iterated elements.
        """
        return len(self._attack_point_iterator_internal)

    def __iter__(self):
        """Start iterating."""
        return iter(self._attack_point_iterator_internal)


class AttackPointIteratorInternalBase(ABC):
    "Attack point iterator abstract class."

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of iterated elements.
        """

    @abstractmethod
    def __iter__(self):
        """Start iterating."""


class AttackPointIteratorInternalConstants(AttackPointIteratorInternalBase):
    """Attack point iterator class that iterates over a constant."""

    def __init__(self, name: str, values: List[List[int]]) -> None:
        """Initialize the constants to iterate."""
        self._name = name
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter({self._name: value} for value in self._values)


class AttackPointIteratorInternalBalancedGenerator(
        AttackPointIteratorInternalBase):
    """
    Attack point iterator class that iterates over the balanced generator.
    """

    def __init__(self, name: str, kwargs: Dict) -> None:
        """Initialize the balanced kwargs to iterate."""
        self._name = name
        self._kwargs = kwargs
        self._len = (self._kwargs.get("bunches", 1) *
                     self._kwargs.get("elements", 256))

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return iter({self._name: value} for value in balanced_generator(
            length=self._kwargs["length"],
            bunches=self._kwargs.get("bunches", 1),
            elements=self._kwargs.get("elements", 256)))


class AttackPointIteratorInternalUnrestrictedGenerator(
        AttackPointIteratorInternalBase):
    """
    Attack point iterator class that iterates over the unrestricted generator.
    """

    def __init__(self, name: str, kwargs: Dict) -> None:
        """Initialize the unrestricted kwargs to iterate."""
        self._name = name
        self._kwargs = kwargs
        self._len = (self._kwargs.get("bunches", 1) *
                     self._kwargs.get("elements", 256))

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return iter({self._name: value} for value in unrestricted_generator(
            length=self._kwargs["length"],
            bunches=self._kwargs.get("bunches", 1),
            elements=self._kwargs.get("elements", 256)))
