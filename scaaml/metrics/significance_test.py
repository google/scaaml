# Copyright 2025 Google LLC
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
"""Custom metrics: MeanRank, MaxRank.

Related metrics:
  MinRank: Is zero as soon as accuracy is non-zero.
  keras.metrics.TopKCategoricalAccuracy: How often the correct class
    is in the top K predictions (how often is rank less than K).
"""
from typing import Any, Optional

import numpy as np
import keras
from keras.metrics import Metric, categorical_accuracy

from scaaml.utils import requires


class SignificanceTest(Metric):  # type: ignore[no-any-unimported,misc]
    """Calculates the probability that a random guess would get the same
    accuracy. Probability is in the interval [0, 1] (impossible to always). By
    convention one rejects the null hypothesis at a given p-value (say 0.005 if
    we want to be sure).

    The method `SignificanceTest.result` requires SciPy to be installed. We
    also mark the `__init__` so that users do not waste time without a chance
    to get the result.

    Args:
      name: (Optional) String name of the metric instance.
      dtype: (Optional) Data type of the metric result.

    Standalone usage:

    ```python
    >>> m = SignificanceTest()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.6, 0.4]])
    >>> m.result().numpy()
    0.25
    ```

    Usage with `compile()` API:

    ```python
    model.compile(optimizer="sgd",
                  loss="mse",
                  metrics=[SignificanceTest()])
    ```
    """

    @requires("scipy")
    def __init__(self, name: str = "SignificanceTest", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.possibilities = self.add_weight(name="possibilities",
                                             initializer="zeros")
        self.seen = self.add_weight(name="seen", initializer="zeros")

    def update_state(self,
                     y_true: Any,
                     y_pred: Any,
                     sample_weight: Optional[Any] = None) -> None:
        """Update the state.

        Args:
          y_true (batch of one-hot): One-hot ground truth values.
          y_pred (batch of one-hot): The prediction values.
          sample_weight (Optional weights): Does not make sense, as we count
            maximum.
        """
        del sample_weight  # unused

        # Make into tensors.
        y_true = np.array(y_true, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=np.float32)

        # Update the number of seen examples.
        self.seen.assign(self.seen + y_true.shape[0])

        # Update the number of correctly predicted examples.
        correct_now = keras.ops.sum(categorical_accuracy(y_true, y_pred))
        self.correct.assign(self.correct + correct_now)

        # Update the number of possibilities.
        self.possibilities.assign(y_true.shape[-1])

    @requires("scipy")
    def result(self) -> Any:
        """Return the result."""

        # Binomial distribution(n, p) -- how many successes out of n trials,
        # each succeeds with probability p independently on others.
        # scipy.stats.binom.cdf(k, n, p) -- probability there are <= k
        # successes.
        # We want to answer what is the probability that a random guess has at
        # least self.correct or more successes. Which is the same as 1 -
        # probability that it has at most k-1 successes.
        k = self.correct.numpy()
        n = self.seen.numpy()
        possibilities = self.possibilities.numpy()
        return 1 - scipy.stats.binom.cdf(  # pylint: disable=undefined-variable
            k - 1,
            n,
            1 / possibilities,
        )

    def reset_state(self) -> None:
        """Reset the state for new measurement."""
        # The state of the metric will be reset at the start of each epoch.
        self.seen.assign(0.0)
        self.correct.assign(0.0)
        self.possibilities.assign(0.0)
