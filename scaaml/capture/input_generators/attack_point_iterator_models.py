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
"""Pydantic models for the attack point iterator."""
import itertools
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Iterator, Literal, List, TypeAlias, Union

from scaaml.capture.input_generators.attack_point_iterator_exceptions import LengthIsInfiniteException, ListNotPrescribedLengthException
from scaaml.capture.input_generators.input_generators import balanced_generator, unrestricted_generator

AttackPointIteratorT = Iterator[Dict[str, Any]]


class ConstantIteratorModel(BaseModel):
    """
    Attack point iterator pydantic model that iterates over given constants.
    
    Args:
        operation (Literal["constants"]): The operation of the iterator
        represents what the iterator does and what 
        arguments should be present.

        name (str): The name represents the key name of the value.

        length (int): The prescribed length for each list in values.
        If one of the lists isn't the same length as this variable
        it will raise an ListNotPrescribedLengthException.

        values (List[List[int]]): List of lists of ints that gets
        iterated through.
    """
    operation: Literal["constants"] = "constants"
    name: str
    length: int
    values: List[List[int]]

    @model_validator(mode="after")
    def check(self) -> "ConstantIteratorModel":
        for value in self.values:
            if len(value) != self.length:
                raise ListNotPrescribedLengthException(
                    f"The prescribed length is {self.length} and \
                    the length of {value} is {len(value)}.")
        return self

    def __len__(self) -> int:
        """Return the number of iterated elements."""
        return len(self.values)

    def items(self) -> AttackPointIteratorT:
        return iter({self.name: value} for value in self.values)

    def get_generated_keys(self) -> List[str]:
        return [self.name]


class GeneratedIteratorModel(BaseModel):
    """
    Attack point iterator pydantic model that iterates over
    the balanced and unrestricted generator.

    operation (Literal["balanced_generator", "unrestricted_generator"]):
        The operation of the iterator
        represents what the iterator does and what 
        arguments should be present.

    name (str): The name represents the key name of the value.
    
    length (int): Length of each yielded array for the generator.

    bunches (int): How many bunches to return (yields `bunches * elements`
        random arrays) for the generator.
    
    elements (int): Each array contains numbers in `range(elements)`
        for the generator.
    """

    operation: Literal["balanced_generator", "unrestricted_generator"]
    name: str
    length: int
    bunches: int = 1
    elements: int = 256

    def __len__(self) -> int:
        """Return the number of iterated elements."""
        return self.bunches * self.elements

    def items(self) -> AttackPointIteratorT:
        """This function returns an Iterator of the items that should be
        iterated through."""
        if self.operation == "balanced_generator":
            generator = balanced_generator
        elif self.operation == "unrestricted_generator":
            generator = unrestricted_generator
        else:
            raise ValueError(f"Unknown generator type: {self.operation}")

        return iter({self.name: value} for value in generator(
            length=self.length, bunches=self.bunches, elements=self.elements))

    def get_generated_keys(self) -> List[str]:
        return [self.name]


BasicIteratorModels: TypeAlias = Union[ConstantIteratorModel,
                                       GeneratedIteratorModel]


class RepeatIteratorModel(BaseModel):
    """
    Initialize the repeated iterate. If repetitions is not present
    or set to a negative number it will do an infinite loop and
    if it is 0 it will not repeat at all.
          
        Args:
            operation (Literal['repeat']): The operation of the iterator
                represents what the iterator does and what 
                has to be in the config file.

            configuration (BasicIteratorModels): The config for the iterated object
                that will get repeated.
                
            repetitions (int): This parameter decides how many times the
                iterator gets repeated. If it is a negative number it
                will repeat infinitely and if you call __len__ it will
                raise an LengthIsInfiniteException. If it is 0 then it will not
                repeat at all. If it is a positive number it will
                repeat that many times.
    """
    operation: Literal["repeat"]
    repetitions: int = Field(default=-1)
    configuration: BasicIteratorModels

    @model_validator(mode="after")
    def check_model(self) -> "RepeatIteratorModel":
        if len(self.configuration) == 0:
            self.repetitions = 0
        return self

    def __len__(self) -> int:
        if self.repetitions >= 0:
            return self.repetitions * len(self.configuration)
        elif len(self.configuration) == 0:
            return 0
        else:
            raise LengthIsInfiniteException("The length is infinite!")

    def items(self) -> AttackPointIteratorT:
        """This function returns an Iterator of the items that should be
        iterated through."""
        if self.repetitions < 0:
            return iter(itertools.cycle(self.configuration.items()))
        return iter(value for _ in range(self.repetitions)
                    for value in self.configuration.items())

    def get_generated_keys(self) -> List[str]:
        """Returns an exhaustive list of names this iterator will create."""
        return self.configuration.get_generated_keys()
