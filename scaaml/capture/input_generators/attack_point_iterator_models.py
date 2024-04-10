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
import collections
import itertools
import math
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Iterator, Literal, List, Tuple, TypeAlias, Union

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
    operation: Literal["constants"] = Field(default="constants")
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
    Attack point iterator pydantic model that iterates over the configuration
    a `repeat` number of times.
          
        Args:
            operation (Literal['repeat']): The operation of the iterator
                represents what the iterator does and what 
                has to be in the config file.

            configuration (BasicIteratorModels): The config for the iterated
                object that will get repeated.
                
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


SimpleIteratorModel: TypeAlias = Union[BasicIteratorModels, RepeatIteratorModel]


class ZipIteratorModel(BaseModel):
    """
    Attack point iterator zip pydantic model. This class takes any amount of
    operands and combines them similar to the `zip` function in Python.
    
    Args:
        operation (Literal['zip']): The operation of the iterator
            represents what the iterator does and what 
            has to be in the config file. This is only used once to
            double check if the operation is the correct one.
            
        operands (List[SimpleIteratorModel]):
            The operands are any number of BasicIteratorModels or
            RepeatIteratorModels that will be combined.
    """
    operation: Literal["zip"]
    operands: List[SimpleIteratorModel]

    def __len__(self) -> int:
        non_negative_lengths: List[int] = []
        for operand in self.operands:
            try:
                if len(operand) >= 0:
                    non_negative_lengths.append(len(operand))
            except LengthIsInfiniteException:
                pass
        if not self.operands:
            return 0
        else:
            # If `non_negative_lengths` is empty it means that all
            # operands are infinite.
            smallest_length: int = min(non_negative_lengths, default=-1)
            if smallest_length < 0:
                raise LengthIsInfiniteException
            return smallest_length

    def items(self) -> AttackPointIteratorT:
        items_of_operands: List[AttackPointIteratorT] = []
        for operand in self.operands:
            items_of_operands.append(operand.items())
        return iter(
            self._merge_dictionaries(tuple_of_dictionaries)
            for tuple_of_dictionaries in zip(*items_of_operands))

    @staticmethod
    def _merge_dictionaries(
        tuple_of_dictionaries: Tuple[Dict[str,
                                          List[int]]]) -> Dict[str, List[int]]:
        merged_dictionary = {}
        for value in tuple_of_dictionaries:
            merged_dictionary.update(value)
        return merged_dictionary

    def get_generated_keys(self) -> List[str]:
        generated_keys = []
        for operand in self.operands:
            generated_keys += operand.get_generated_keys()
        return generated_keys


class CartesianProductIteratorModel(BaseModel):
    """
    Attack point iterator cartesian product pydantic model. This class takes
    any amount of operands and combines them just like a cartesian product 
    would.

    Args:
        operation (Literal['cartesian_product']): The operation of the iterator
            represents what the iterator does and what 
            has to be in the config file. This is only used once to
            double check if the operation is the correct one.
            
        operands (List[Union[SimpleIteratorModel, "ComplicatedIteratorModel"]]):
           The operands are any number of BasicIteratorModels or
            RepeatIteratorModels that will be combined. If the operands list
            is empty it will raise a ValueError. If one of the operands
            length is 0 the length of the cartesian product iterator will
            also be 0, it will return an empty iterator. If one of the
            operands iterates infinitely it will throw a
            LengthIsInfiniteException in the init.
    """
    operation: Literal["cartesian_product"]
    operands: List[Union[SimpleIteratorModel, "ComplicatedIteratorModel"]]

    @model_validator(mode="after")
    def check_model(self) -> "CartesianProductIteratorModel":
        """If the length of the operands is bigger then 1 it will set operands
        to the first item of the operands and a CartesianProductIteratorModel
        of the remaining list of operands as the second item."""
        if len(self.operands) > 1:
            self.operands = [
                self.operands[0],
                CartesianProductIteratorModel(operation="cartesian_product",
                                              operands=self.operands[1:])
            ]
        return self

    def __len__(self) -> int:
        operand_lengths: List[int] = []
        for operand in self.operands:
            try:
                operand_lengths.append(len(operand))
            except LengthIsInfiniteException:
                operand_lengths.append(-1)

        if any(length == 0 for length in operand_lengths):
            return 0
        elif any(length < 0 for length in operand_lengths):
            raise LengthIsInfiniteException
        else:
            return math.prod(operand_lengths)

    def items(self) -> AttackPointIteratorT:
        try:
            if len(self) == 0:
                return iter([])
        except LengthIsInfiniteException:
            pass

        if len(self.operands) == 2:
            return iter({
                **value_one,
                **value_two
            }
                        for value_one in self.operands[0].items()
                        for value_two in self.operands[1].items())
        else:
            return iter(self.operands[0].items())

    def get_generated_keys(self) -> List[str]:
        generated_keys: List[str] = []
        for operand in self.operands:
            generated_keys += operand.get_generated_keys()
        return generated_keys


ComplicatedIteratorModel: TypeAlias = Union[ZipIteratorModel,
                                            CartesianProductIteratorModel]


class IteratorModel(BaseModel):
    """
    This is the general iterator pydantic model which combines all of the other
    models into one. With IteratorModel.validate_model(dict) the user can
    create an instance with a dict. To see how the dict should look like
    use IteratorModel.model_json_schema().

    Args:
        iterator_model (Union[BasicIteratorModels, ComplicatedIteratorModel,
        RepeatIteratorModel]): The iterator model combines all of the models
        together.
            
    """
    iterator_model: Union[BasicIteratorModels, ComplicatedIteratorModel,
                          RepeatIteratorModel]

    @model_validator(mode="after")
    def check_duplicate_names(self) -> "IteratorModel":
        # Check that all names are unique
        names_list = collections.Counter(
            self.iterator_model.get_generated_keys())
        duplicates = [name for name, count in names_list.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicated attack point names {duplicates}")
        return self

    def __len__(self) -> int:
        return len(self.iterator_model)

    def items(self) -> AttackPointIteratorT:
        return self.iterator_model.items()
