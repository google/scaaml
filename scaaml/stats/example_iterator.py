"""Iterates over examples in a dataset."""

from typing import Union

from scaaml.io import Dataset


class ExampleIterator:
    """Iterate through examples in a dataset. Examples are returned as if
    iterating using scaaml.io.Dataset.inspect(...).as_numpy_iterator(), that
    is a dictionary of attack points and traces.

    Example use:
      for example in ExampleIterator(dataset_path='ds_folder', split='test'):
          print(example['key'])
    """
    def __init__(self,
                 dataset_path: str,
                 split: Union[None, str] = None,
                 group: Union[None, int] = None,
                 part: Union[None, int] = None) -> None:
        """Example iterator.

        Args:
          dataset_path: Path to the dataset.
          split: If None, then all splits are iterated. Otherwise only one
            split is iterated (one of 'train', 'test', 'holdout', for more
            info see scaaml.io.Dataset.SPLITS).
          group: If None, then all groups are iterated. Otherwise only shards
            belonging to this group.
          part: If None, then all parts are iterated. Otherwise only shards
            belonging to this part.
        """
        self._dataset_path = dataset_path
        if split is None:
            splits = Dataset.SPLITS
        else:
            splits = [split]
        self._shards_list = []
        dataset = Dataset.from_config(dataset_path=dataset_path, verbose=False)
        for current_split in splits:
            for shard_id, shard in enumerate(
                    dataset.shards_list[current_split]):
                if group is not None and shard['group'] != group:
                    continue  # Skip this shard.
                if part is not None and shard['part'] != part:
                    continue  # Skip this shard.
                self._shards_list.append((shard_id, shard, current_split))
        self._shard_idx = 0
        # _shard_iterator is initialized by the first call of __next__.
        self._shard_iterator = iter([])

    def __iter__(self):
        """Returns self."""
        return self

    def __next__(self):
        """Return next example.

        Returns: The next example as returned by
          next(Dataset.inspect(...).as_numpy_iterator()).

        Raises: StopIteration if there is no example to iterate.
        """
        try:
            return next(self._shard_iterator)
        except StopIteration:
            if self._shard_idx >= len(self._shards_list):
                raise StopIteration
            shard_id = self._shards_list[self._shard_idx][0]
            shard = self._shards_list[self._shard_idx][1]
            split = self._shards_list[self._shard_idx][2]
            self._shard_idx += 1
            num_example = shard['examples']
            self._shard_iterator = Dataset.inspect(
                dataset_path=self._dataset_path,
                split=split,
                shard_id=shard_id,
                num_example=num_example,
                verbose=False).as_numpy_iterator()
            return next(self)

    def __len__(self) -> int:
        """Returns the length. Assumes that there are the same number of
        examples in each shard.

        Returns: The number of examples that are iterated.
        """
        if not self._shards_list:
            return 0
        n_shards = len(self._shards_list)
        n_examples = self._shards_list[0][1]['examples']
        return n_shards * n_examples
