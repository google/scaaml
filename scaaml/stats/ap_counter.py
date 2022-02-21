"""Counts how many times each value of each attack point appears."""

from typing import Dict, List, Union

import numpy as np


class APCounter:
    """APCounter counts how many times does a given attack point attain
    concrete value (for instance how many times does the fifth byte of key has
    value 10).  These counts are reported by get_counts as a numpy array of
    shape (len, max_val). Can be used with ExampleIterator or directly while
    capturing a dataset.

    Example use:
    counter = APCounter({'len': 16, 'max_val': 256})
    counter.update([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    counts = counter.get_counts()  # counts[i][i] == 1 for i in range(16)
    """
    def __init__(self, attack_point_info: Dict[str, int]) -> None:
        """Create a new counter.

        Args:
          attack_point_info: Information about this attack point. It is
            assumed that this dictionary contains 'len' and 'max_val' (both
            natural numbers).
        """
        self._len = attack_point_info['len']
        self._max_val = attack_point_info['max_val']
        self._cnt = np.zeros((self._len, self._max_val), dtype=np.int64)
        self._byte_numbers = np.arange(self._len)

    def update(self, attack_point: List[int]) -> None:
        """Update count with a single list of attack points. See also
        APCounter.update_one_hot.

        Args:
          attack_point: List of attack_point_info['len'] integers from
            range(attack_point_info['max_val']).

        Example use:
          # The attack point has four bytes (each with a value between 0 and
          # max_val-1). This example had the first byte of value zero, second
          # of value 255, third of 1 and fourth of 2.
          ap_counter.update([0, 255, 1, 2])
        """
        assert len(attack_point) == self._len
        self._cnt[self._byte_numbers, attack_point] += 1

    def update_one_hot(self, attack_point: List[List]) -> None:
        """Convenience alternative of update, when the attack point is one-hot
        encoded.
        """
        raise NotImplementedError('TODO(karelkral): Implement update_one_hot')

    def get_counts(self, byte: Union[None, int] = None) -> np.ndarray:
        """Return the counts. If byte is specified returns a one-dimensional
        array of integers of length max_val. If byte is None returns a
        two-dimensional array of shape (len, max_val).

        Args:
          byte: If None return values for all bytes. If an integer from
            range(len), then return counts for the specific values only.

        Returns: Array of counts.
        """
        if byte is None:
            return self._cnt
        return self._cnt[byte]
