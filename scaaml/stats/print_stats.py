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
"""Print results of statistical checks and observations. Iterates only once
over the dataset. Implemented as a class to allow use during capture as well as
standalone."""

from __future__ import annotations
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Set, cast, get_args

import numpy as np
import numpy.typing as npt

from tabulate import tabulate
from tqdm import tqdm

from scaaml.io import Dataset
from scaaml.stats.ap_counter import APCounter
from scaaml.stats.ap_checker import APChecker
from scaaml.stats.example_iterator import ExampleIterator
from scaaml.stats.trace_stddev_of_stat import STDDEVofAVGofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofMAXofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofMINofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofSTATofTraces


def _class_name(obj: object) -> str:
    """Return the class name of the given object."""
    return obj.__class__.__name__


class PrintStats:
    """Gather and print warnings from attack point checks and trace
    statistics.

    Example use:

    # On an existing dataset:
    print_stats = PrintStats.from_config(dataset_path="/mnt/path/to/dataset")

    # During capture:
    print_stats = PrintStats(measurements_info, attack_points_info)
    for example in example_iterator:
        print_stats.add_example(example=example,
                                split=split,
                                group=group,
                                part=part)
    print_stats.print()
    """

    def __init__(self, measurements_info: Dict[str, Any],
                 attack_points_info: Dict[str, Dict[str, int]]) -> None:
        self._measurements_info = measurements_info
        self._attack_point_info = attack_points_info
        self._split_ap_counters = {
            split: {
                ap_name: [APCounter(attack_point_info=ap_info),]
                for ap_name, ap_info in attack_points_info.items()
            } for split in Dataset.SPLITS
        }

        self._split_trace_stats = {
            split: {
                trace_name: self.get_all_trace_stats()
                for trace_name in measurements_info
            } for split in Dataset.SPLITS
        }
        self._split_trace_group_stats: Dict[str, Dict[str, Dict[
            int, List[STDDEVofSTATofTraces]]]] = {
                split: {
                    trace_name: defaultdict(self.get_all_trace_stats)
                    for trace_name in measurements_info
                } for split in Dataset.SPLITS
            }
        self._all_groups: Set[int] = set()

    @staticmethod
    def get_all_trace_stats() -> List[STDDEVofSTATofTraces]:
        # Add trace statistics here.
        return [
            STDDEVofAVGofTraces(),
            STDDEVofMAXofTraces(),
            STDDEVofMINofTraces(),
        ]

    @staticmethod
    def from_config(dataset_path: str) -> PrintStats:
        """Read a dataset and print all attack point check warnings and trace
        statistics.

        Args:
          dataset_path: Root path of the dataset.
        """
        dataset = Dataset.from_config(dataset_path)
        print_stats = PrintStats(measurements_info=dataset.measurements_info,
                                 attack_points_info=dataset.attack_points_info)
        all_splits = set(dataset.shards_list.keys())
        all_groups = set(shard["group"]
                         for split in Dataset.SPLITS
                         for shard in dataset.shards_list[split])
        all_parts = set(shard["part"]
                        for split in Dataset.SPLITS
                        for shard in dataset.shards_list[split])
        prod = list(product(all_splits, all_groups, all_parts))
        for split, group, part in tqdm(prod):
            assert split in get_args(Dataset.SPLIT_T)
            split = cast(Dataset.SPLIT_T, split)
            example_iterator = ExampleIterator(dataset_path=dataset_path,
                                               split=split,
                                               group=group,
                                               part=part)
            for example in tqdm(example_iterator, leave=False):
                print_stats.add_example(example=example,
                                        split=split,
                                        group=group,
                                        part=part)
        print_stats.print()
        return print_stats

    def add_example(self, example: Dict[str, npt.NDArray[np.generic]],
                    split: Dataset.SPLIT_T, group: int, part: int) -> None:
        """Add another example.

        Args:
          example: A dictionary containing attack points and measurements
            (traces).
          split: The split this example belongs to.
          group: The group this example belongs to.
          part: The part this example belongs to.
        """
        # Part is unused now, but is there when we need more granularity.
        _ = part
        self._all_groups.add(group)
        for ap_name in self._attack_point_info:
            for ap_counter in self._split_ap_counters[split][ap_name]:
                ap_counter.update(
                    attack_point=cast(List[int], example[ap_name]))
        for trace_name in self._measurements_info:
            for trace_stat in self._split_trace_stats[split][trace_name]:
                trace_stat.update(trace=example[trace_name])
            for trace_stat in self._split_trace_group_stats[split][trace_name][
                    group]:
                trace_stat.update(trace=example[trace_name])

    def _print_attack_point_warnings(self) -> None:
        """Print warnings for attack points."""
        print("Attack point check warnings:")
        for split in Dataset.SPLITS:
            print("")
            print(split)
            for ap_name in self._attack_point_info:
                for ap_count in self._split_ap_counters[split][ap_name]:
                    # Add attack point checks here.
                    _ = APChecker(counts=ap_count.get_counts(),
                                  attack_point_name=ap_name)

    def _print_trace_statistics(self) -> None:
        """Print table of statistics. Rows are statistics, columns are splits.
        """
        print("Trace statistics:")
        for trace_name in self._measurements_info:
            print("")
            table: List[List[str]] = [[f"{trace_name} stats", *Dataset.SPLITS]]
            # Fill the statistic names.
            for trace_stat in self.get_all_trace_stats():
                trace_stat_name = _class_name(trace_stat)
                table.append([trace_stat_name, *["0" for _ in Dataset.SPLITS]])
            # Fill the values.
            for i, split in enumerate(Dataset.SPLITS):
                # Check that we are filling the right column.
                assert split == table[0][i + 1]
                for j, trace_stat in enumerate(
                        self._split_trace_stats[split][trace_name]):
                    # Check that we are filling the right row.
                    assert _class_name(trace_stat) == table[j + 1][0]
                    table[j + 1][i + 1] = str(trace_stat.result())
            print(tabulate(table, headers="firstrow"))

    def _print_trace_statistics_per_group(self) -> None:
        """Print a table of results for each statistics. Rows are groups,
        columns are splits."""
        print("Trace statistics per group:")
        for trace_name in self._measurements_info:
            for i, trace_stat in enumerate(self.get_all_trace_stats()):
                print("")
                print(f"{trace_name} {_class_name(trace_stat)}:")
                table: List[List[str]] = [["group", *Dataset.SPLITS]]
                for group in self._all_groups:
                    table.append([str(group), *["NA" for _ in Dataset.SPLITS]])
                    for j, split in enumerate(Dataset.SPLITS):
                        if group in self._split_trace_group_stats[split][
                                trace_name]:
                            curr_trace_stat = self._split_trace_group_stats[
                                split][trace_name][group][i]
                            # Check that we are filling the right table.
                            assert _class_name(curr_trace_stat) == _class_name(
                                trace_stat)
                            table[-1][j + 1] = str(curr_trace_stat.result())
                print(tabulate(table, headers="firstrow"))

    @staticmethod
    def _print_separator() -> None:
        """Print separator between two blocks of warnings / statistics."""
        print("")
        print("-------------------------------------------")
        print("")

    def print(self) -> None:
        """Print warnings from attack point checkers and statistics about
        traces."""
        self._print_attack_point_warnings()
        PrintStats._print_separator()
        self._print_trace_statistics()
        PrintStats._print_separator()
        self._print_trace_statistics_per_group()
