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
from keras.metrics import Metric, MeanMetricWrapper, categorical_accuracy


class SignificanceTest(Metric):  # type: ignore[no-any-unimported,misc]
    """Calculates the probability that a random guess would get the same
    accuracy. Probability is in the interval [0, 1] (impossible to always). By
    convention one rejects the null hypothesis at a given p-value (say 0.005 if
    we want to be sure).

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

    def __init__(self, name: str = "SignificanceTest", **kwargs: Any) -> None:
        raise ImportError("To use the SignificanceTest please install scipy")

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
        raise ImportError("To use the SignificanceTest please install scipy")

    def result(self) -> Any:
        """Return the result."""
        raise ImportError("To use the SignificanceTest please install scipy")

    def reset_state(self) -> None:
        """Reset the state for new measurement."""
        raise ImportError("To use the SignificanceTest please install scipy")
