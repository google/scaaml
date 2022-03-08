"""Print results of statistical checks and observations. Iterates only once
over the dataset. Implemented as a class to allow use during capture as well as
standalone."""

from itertools import product
from typing import Dict

from tabulate import tabulate
from tqdm import tqdm

from scaaml.io import Dataset
from .ap_counter import APCounter
from .ap_checker import APChecker
from .example_iterator import ExampleIterator
from .trace_stddev_of_avg import STDDEVofAVGofTraces


class PrintStats:
    """Gather and print warnings from attack point checks and trace
    statistics.

    Example use:

    # On an existing dataset:
    print_stats = PrintStats.from_config(dataset_path='/mnt/path/to/dataset')

    # During capture:
    print_stats = PrintStats(measurements_info, attack_points_info)
    for example in example_iterator:
        print_stats.add_example(example=example,
                                split=split,
                                group=group,
                                part=part)
    print_stats.print()
    """
    def __init__(self, measurements_info: Dict,
                 attack_points_info: Dict) -> None:
        self._measurements_info = measurements_info
        self._attack_point_info = attack_points_info
        self._split_ap_counters = {
            split: {
                ap_name: [
                    APCounter(attack_point_info=ap_info),
                ]
                for ap_name, ap_info in attack_points_info.items()
            }
            for split in Dataset.SPLITS
        }
        self._split_trace_stats = {
            split: {
                trace_name: [
                    STDDEVofAVGofTraces(),
                ]
                for trace_name in measurements_info
            }
            for split in Dataset.SPLITS
        }

    @staticmethod
    def from_config(dataset_path: str) -> None:
        """Read a dataset and print all attack point check warnings and trace
        statistics.

        Args:
          dataset_path: Root path of the dataset.
        """
        dataset = Dataset.from_config(dataset_path)
        print_stats = PrintStats(measurements_info=dataset.measurements_info,
                                 attack_points_info=dataset.attack_points_info)
        all_splits = set(dataset.shards_list.keys())
        all_groups = set(shard['group'] for split in Dataset.SPLITS
                         for shard in dataset.shards_list[split])
        all_parts = set(shard['part'] for split in Dataset.SPLITS
                        for shard in dataset.shards_list[split])
        prod = list(product(all_splits, all_groups, all_parts))
        for split, group, part in tqdm(prod):
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

    def add_example(self, example: Dict, split: str, group: int,
                    part: int) -> None:
        """Add another example.

        Args:
          example: A dictionary containing attack points and measurements
            (traces).
          split: The split this example belongs to.
          group: The group this example belongs to.
          part: The part this example belongs to.
        """
        # Group and part are unused now, but are there when we need more
        # granularity.
        _ = group
        _ = part
        for ap_name in self._attack_point_info:
            for ap_counter in self._split_ap_counters[split][ap_name]:
                ap_counter.update(attack_point=example[ap_name])
        for trace_name in self._measurements_info:
            for trace_stat in self._split_trace_stats[split][trace_name]:
                trace_stat.update(trace=example[trace_name])

    def print(self) -> None:
        """Print warnings from attack point checkers and statistics about
        traces."""
        print('Attack point check warnings:')
        for split in Dataset.SPLITS:
            print('')
            print(split)
            for ap_name in self._attack_point_info:
                for ap_count in self._split_ap_counters[split][ap_name]:
                    _ = APChecker(counts=ap_count.get_counts(),
                                  attack_point_name=ap_name)

        print('')
        print('-------------------------------------------')
        print('')

        print('Trace statistics:')
        for trace_name in self._measurements_info:
            print('')
            table = [[f'{trace_name} stats', *Dataset.SPLITS]]
            # Fill the statistic names.
            for trace_stat in self._split_trace_stats[
                    Dataset.SPLITS[0]][trace_name]:
                trace_stat_name = trace_stat.__class__.__name__
                table.append([trace_stat_name, *[0 for _ in Dataset.SPLITS]])
            # Fill the values.
            for i, split in enumerate(Dataset.SPLITS):
                # Check that we are filling the right column.
                assert split == table[0][i + 1]
                for j, trace_stat in enumerate(
                        self._split_trace_stats[split][trace_name]):
                    # Check that we are filling the right row.
                    assert trace_stat.__class__.__name__ == table[j + 1][0]
                    table[j + 1][i + 1] = trace_stat.result()
        print(tabulate(table, headers='firstrow'))
