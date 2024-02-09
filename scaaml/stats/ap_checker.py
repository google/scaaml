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
"""Runs checks and reports failures. A failed test does not mean an error in
the dataset, interpretation is problem specific."""

from typing import Callable

import numpy as np
import numpy.typing as npt
from pprint import pprint


class APChecker:
    """Checks that the attack point is random enough based on counts.

    Args:
      counts: Numpy array of counts of shape (len, max_val) where
        counts[i][val] counts how many times does the i-th piece (e.g., byte
        or bit) attain value val.
      attack_point_name: The name of the attack point. Useful for debugging
        purposes.

    Example use:
    km_checker = APChecker(counts=km_counter.get_counts(),
                           attack_point_name='km')
    """

    def __init__(self, counts: npt.NDArray[np.int64],
                 attack_point_name: str) -> None:
        self._counts = counts.copy()
        self._attack_point_name = attack_point_name
        self._something_failed = False
        self.run_all()
        # If any of those checks failed, print the counts.
        if self._something_failed:
            pprint(self._counts)

    def run_all(self) -> None:
        """Run all statistical checks. When adding a new test remember to call
        it from this method. To test that your method is called from run_all,
        take a look at
        tests/stats/test_ap_checker.py::test_run_all_calls_check_all_nonzero.

        Raises: Does not re-raise ValueError, re-raises all other types of
        errors. If any method here raises ValueError, prints the error and
        after all errors prints the counts.
        """
        self._run_check(self.check_all_nonzero)

    def _run_check(self, check: Callable[[], None]) -> None:
        """Run the given check, if it raises ValueError change take a note
        and print the error message.

        Args:
          check: The member method to call.

        Example usage: See APChecker.run_all.

        Raises: Does not re-raise ValueError, re-raises all other types of
        errors.
        """
        try:
            check()
        except ValueError as value_error:
            self._something_failed = True
            print(value_error)

    @property
    def attack_point_name(self) -> str:
        """Returns the name of the attack point."""
        return self._attack_point_name

    def check_all_nonzero(self) -> None:
        """Check that every value of the attack point appears at least once.

        Raises: ValueError if there is a value of an attack point that is
          never present.
        """
        # The ap_piece is either byte or bit of an attack point (e.g. single
        # byte of the key).
        for ap_piece_number, ap_piece in enumerate(self._counts):
            if not (ap_piece > 0).all():
                msg = (
                    f'Not all combinations of attack_point-value appear, for '
                    f'example {self._attack_point_name}_{ap_piece_number} '
                    f'never has value {(ap_piece > 0).argmin()}.')
                raise ValueError(msg)
