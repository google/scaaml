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
"""Online computation of statistical values."""

import numpy as np


class Sum:
    """Compute a sum in an online fashion in online fashion, numerically
    stable. Using Neumaier variant of Kahan-Babuska algorithm
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm.

    Example use:
    ```
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = Sum()
    for e in a:
      s.update(e)
    # Print maximum absolute difference from the NumPy implementation.
    print((s.result - a.sum(axis=0)).abs().max())
    ```
    """

    def __init__(self, cast_to: np.dtype = np.float64) -> None:
        """Initialize a sum.

        Args:

          cast_to (np.dtype): Cast each new element to this dtype. Defaults to
          casting to np.float64.
        """
        self._dtype: np.dtype = cast_to
        # Running sum.
        self._sum: np.ndarray | None = None
        # A running compensation for lost low-order bits. In infinite precision
        # this would be equal to zero.
        self._c: np.ndarray | None = None

    @property
    def result(self) -> np.ndarray | None:
        """Return the result (here the sum). Or None if no summation was done
        (unknown shape).
        """
        # Beware of `if not self._sum` due to zero.
        if self._sum is None:
            # Either one or both are None.
            assert self._c is None
            return None

        return self._sum + self._c

    def update(self, value: np.ndarray) -> None:
        """Update by one element.

        Args:

          value (np.ndarray): Next value to be summed.
        """
        # Beware of `if not self._sum` due to zero.
        if self._sum is None:
            # First element.
            self._sum = np.zeros_like(value, dtype=self._dtype)
            self._c = np.zeros_like(value, dtype=self._dtype)

        t = self._sum + value

        # Add compensation for the low order bits which would be lost:
        # - if self._sum is larger then low order bits of value would be lost
        # - if value is larger then vice-versa
        self._c += np.where(
            np.abs(self._sum) >= np.abs(value),  # condition
            (self._sum - t) + value,  # value when condition is true
            (value - t) + self._sum,  # value when condition is false
        )

        # Update the running sum.
        self._sum = t


class Mean:
    """Compute mean in an online fashion.
    """

    def __init__(self, cast_to: np.dtype = np.float64) -> None:
        """Initialize the computation.

        Args:

          cast_to (np.dtype): Cast each new element to this dtype. Defaults to
          casting to np.float64.
        """
        # Seen elements.
        self._elements: int = 0
        self._sum: Sum = Sum(cast_to=cast_to)

    def update(self, value: np.ndarray) -> None:
        """Update by the next value."""
        self._elements += 1
        self._sum.update(value)

    @property
    def result(self) -> np.ndarray | None:
        """Return the arithmetic mean if some values have been added using the
        `update` method. Else return None.
        """
        if self._elements:
            return self._sum.result / self._elements
        return None


class VarianceSinglePass:
    """Welford's algorithm to compute variance. Could be a little slower (CPU)
    since each iteration contains a division and multiplication.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Example use:
    ```
    variance = VarianceSinglePass()

    for value in all_values:
      variance.update(value)

    print((variance.result - all_values.var(axis=0)).abs().max())
    ```
    """

    def __init__(self, cast_to: np.dtype = np.float64, ddof: int = 0) -> None:
        """Initialize a sum.

        Args:

          cast_to (np.dtype): Cast each new element to this dtype. Defaults to
          casting to np.float64.

          ddof (int): Degrees of freedom. Defaults to 0. For Bessel's
          correlation set to 1. For more information see `np.var`.
        """
        self._dtype: np.dtype = cast_to
        self._ddof: int = ddof
        self._mean: Sum = Sum(cast_to=cast_to)
        self._msq: Sum = Sum(cast_to=cast_to)
        self._n_seen: int = 0

    @property
    def n_seen(self) -> int:
        """How many examples have been seen.
        """
        return self._n_seen

    def update(self, value: np.ndarray) -> None:
        """Update by one element.

        Args:

          value (np.ndarray): Next value to be used.
        """
        self._n_seen += 1
        if self._n_seen == 1:
            self._mean.update(value)
            self._msq.update(np.zeros_like(value, dtype=self._dtype))
            return

        delta = value - self._mean.result
        self._mean.update(delta / self._n_seen)
        self._msq.update(delta * (value - self._mean.result))

    @property
    def result(self) -> np.ndarray | None:
        """Return the result (variance). Or None if not enough data seen.
        """
        if self._n_seen < 2:
            return None

        return self._msq.result / (self._n_seen - self._ddof)


class VarianceTwoPass:
    """Two pass algorithm to compute variance. Should be numerically stable and
    fast if iteration is fast.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Example use:
    ```
    variance = VarianceTwoPass()

    for value in all_values:
      variance.update(value)

    variance.set_second_pass()

    for value in all_values:
      variance.update(value)

    print((variance.result - all_values.var(axis=0)).abs().max())
    ```
    """

    def __init__(self, cast_to: np.dtype = np.float64, ddof: int = 0) -> None:
        """Initialize a sum.

        Args:

          cast_to (np.dtype): Cast each new element to this dtype. Defaults to
          casting to np.float64.

          ddof (int): Degrees of freedom. Defaults to 0. For Bessel's
          correlation set to 1. For more information see `np.var`.
        """
        self._dtype: np.dtype = cast_to
        self._ddof: int = ddof
        self._first_iteration: bool = True
        self._n_first_iteration: int = 0
        self._n_second_iteration: int = 0
        self._mean: Mean = Mean(cast_to=cast_to)
        self._sum_diff_square: Sum = Sum(cast_to=cast_to)

    @property
    def result(self) -> np.ndarray | None:
        """Return the result (variance). Or None if not enough data seen.
        """
        if self._n_second_iteration == 0:
            return None
        if self._n_first_iteration != self._n_second_iteration:
            return None

        return self._sum_diff_square.result / (self._n_first_iteration -
                                               self._ddof)

    def set_second_pass(self) -> None:
        """Start second iteration.
        """
        self._first_iteration = False

        self._cached_mean_result: np.ndarray = self._mean.result

    def update(self, value: np.ndarray) -> None:
        """Update by one element.

        Args:

          value (np.ndarray): Next value to be used.
        """
        if self._first_iteration:
            self._n_first_iteration += 1
            self._mean.update(value)
        else:
            self._n_second_iteration += 1
            self._sum_diff_square.update((value - self._cached_mean_result)**2)

    @property
    def n_seen(self) -> int | None:
        """How many examples have been seen or None if the first and second
        iteration were different length.
        """
        if self._n_second_iteration != self._n_first_iteration:
            return None

        return self._n_second_iteration
