# Copyright 2022-2024 Google LLC
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
"""A context manager dealing with opening new shards, part number and group
number."""

from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from scaaml.io.dataset import Dataset


class _DatasetFillerContext:
    """The actual context. Use DatasetFiller which closes the last shard on
    exit. Takes care of creating new shards with proper part_number and
    group_number.

    Args:
      dataset (scaaml.io.Dataset): The dataset object to add examples into.
      examples_per_shard (int): Number of examples in a shard.
      plaintexts_per_key (int): Number of plaintexts used with a concrete key.
      repetitions (int): Number of examples belonging to the same key,
        plaintext pair.
      skip_examples (int): How many examples to skip (non-negative integer
        divisible by examples_per_shard). Defaults to zero.
    """

    def __init__(self,
                 dataset: Dataset,
                 plaintexts_per_key: int,
                 repetitions: int,
                 skip_examples: int = 0) -> None:
        self._dataset = dataset

        # Constants.
        self._examples_per_shard: int = dataset.examples_per_shard
        self._plaintexts_per_key: int = plaintexts_per_key
        self._repetitions: int = repetitions

        # Check that the skip_examples value is valid.
        if skip_examples % self._examples_per_shard:
            raise ValueError(f"{skip_examples} is not divisible by "
                             f"{self._examples_per_shard}")
        if skip_examples < 0:
            raise ValueError(f"{skip_examples} cannot be negative")
        # How many examples have been already written (including the skipped
        # examples).
        self._written_examples: int = skip_examples
        # How many examples have been actually written (excluding the skipped
        # examples). This is used to determine if an example has been written
        # and if we should close the last shard.
        self.written_examples_not_skipped: int = 0

    def write_example(self, *, attack_points: Dict[str, bytearray],
                      measurement: Dict[str, Any], current_key: List[int],
                      split_name: str, chip_id: int) -> None:
        """Write an example. Opens a new shard if necessary.

        Args:
          attack_points (Dict): The attack point values (values of the bytes
            or bits).
          measurement (Dict): Measurements (trace name and trace).
          current_key (List): Bytes of the current key (used in the shard name).
          split_name (str): The split, see scaaml.io.Dataset.SPLITS.
          chip_id (int): Id of the target chip.
        """
        # Open a new shard if necessary.
        if self._need_to_open_new_shard():
            self._dataset.new_shard(
                key=current_key,
                part=self.part_number,
                group=self.group_number,
                split=split_name,
                chip_id=chip_id,
            )

        # Write the current example.
        self._dataset.write_example(
            attack_points=attack_points,
            measurement=measurement,
        )

        # Update counters.
        self._update_counters()

    def _need_to_open_new_shard(self) -> bool:
        """Determine if a new shard needs to be opened."""
        # There were number of examples divisible by the number of examples in
        # a single shard.
        return self._written_examples % self._examples_per_shard == 0

    @property
    def part_number(self) -> int:
        """A part is the id of the shard of a single key."""
        # How many examples with the same key.
        examples_with_same_key = self._plaintexts_per_key * self._repetitions
        # How many examples have been already written in this part.
        examples_in_this_part = self._written_examples % examples_with_same_key
        # Id of the current shard.
        return examples_in_this_part // self._examples_per_shard

    @property
    def group_number(self) -> int:
        """A group is 1 full rotation of the key bytes (0 - 255)."""
        # How many examples are there in a group (256 different byte values ==
        # 256 different keys).
        examples_per_group = 256 * self._plaintexts_per_key * self._repetitions
        return self._written_examples // examples_per_group

    def _update_counters(self) -> None:
        """Update inner counters of part_number and group_number."""
        # Update number of examples in the dataset (i.e., including the skipped
        # examples).
        self._written_examples += 1
        # Update number of examples added using write_example (i.e., excluding
        # the skipped examples).
        self.written_examples_not_skipped += 1

    def close_shard(self) -> None:
        """Close shard. Called automatically by DatasetFiller.__exit__."""
        self._dataset.close_shard()


class DatasetFiller:
    """Takes care of creating new shards with the correct part_number and
    group_number.

    Args:
      dataset (scaaml.io.Dataset): The dataset object to add examples into.
      plaintexts_per_key (int): Number of plaintexts used with a concrete key.
      repetitions (int): Number of examples belonging to the same key,
        plaintext pair.
      skip_examples (int): How many examples to skip (non-negative integer
        divisible by examples_per_shard). Defaults to zero.

    Example use:
    # Context manager properly opens and closes shards.
    with DatasetFiller(
        dataset=dataset,
        plaintexts_per_key=plaintexts_per_key,
        repetitions=repetitions,
        skip_examples=n_examples_to_skip,
    ) as dataset_filler:
        # Add examples, new shards are opened automatically.
        for attack_points, measurement in examples:
            dataset_filler.write_example(
                attack_points=attack_points,
                measurement=measurement,
                current_key=current_key,
                split_name=split_name,
                chip_id=chip_id,
            )
    """

    def __init__(self,
                 dataset: Dataset,
                 plaintexts_per_key: int,
                 repetitions: int,
                 skip_examples: int = 0) -> None:
        self._dataset_filler_context = _DatasetFillerContext(
            dataset=dataset,
            plaintexts_per_key=plaintexts_per_key,
            repetitions=repetitions,
            skip_examples=skip_examples,
        )

    def __enter__(self) -> _DatasetFillerContext:
        """Initialize _DatasetFillerContext."""
        return self._dataset_filler_context

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Make sure to close the last shard.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        # Close the shard only if there was an example written.
        if self._dataset_filler_context.written_examples_not_skipped > 0:
            self._dataset_filler_context.close_shard()
