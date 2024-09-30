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
import scipy


def rank(y_true: Any, y_pred: Any, optimistic: bool = False) -> Any:
    """Calculates the rank of the correct class (counted from 0). If the
    prediction is correct, the rank is equal to zero. If the correct class is
    the least probable according to y_pred, than the rank is equal to number
    of classes - 1.

    When there is a tie in y_pred (two or more classes are assigned the same
    probability) then the correct class is assumed to be the least probable.
    Formally rank of a target class is the number of classes that have same or
    higher probability - 1. If the parameter 'optimistic' is True then number
    of classes with strictly higher probability is returned. When all
    probabilities are different the parameter 'optimistic' plays no role.

    You can provide logits of classes as `y_pred`, since argmax of logits and
    probabilities are same.

    Standalone usage:
    >>> y_true = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.], [0.5, 0.5, 0.]]
    >>> r = rank(y_true, y_pred)
    >>> assert r.shape == (3,)
    >>> r.numpy()
    array([1., 0., 1.], dtype=float32)

    Args:
      y_true (batch of one-hot): One-hot ground truth values.
      y_pred (batch of probabilities): The prediction values.
      optimistic (bool): If True then ties are decided in favor of target.
        Defaults to False.

    Returns:
      Rank values.
    """
    # Let N be the number of classes (N=256 when predicting a single byte).
    # y_true, y_pred are of shape (None, N) (the None stands for a yet
    # unknown batch size).
    #
    # We count how many classes have at least as high probability as the
    # correct class and subtract 1.
    #
    # mul[argmax(y_true)] == y_pred[argmax(y_true)] and zero elsewhere.
    mul = keras.ops.multiply(y_true, y_pred)
    # mul is of shape (None, N)

    # We get the predicted probability for the correct class (keeping the
    # dimension, so we can compare with y_pred).
    predicted_pr = keras.ops.max(mul, axis=-1, keepdims=True)
    # predicted_pr is of shape (None, 1)

    # Boolean array where y_pred >= predicted_pr (True values denote the
    # classes that are assigned at least as high probability as the correct
    # class). If we are optimistic we count only y_pred > predicted_pr (break
    # ties in favor of the target).
    if optimistic:
        preferred = keras.ops.greater(y_pred, predicted_pr)
    else:
        preferred = keras.ops.greater_equal(y_pred, predicted_pr)
    # preferred is of shape (None, N)

    # A scalar, summing a 1. for each True and 0. for each False.
    ranks = keras.ops.sum(keras.ops.cast(preferred, "float32"), axis=-1)
    # ranks is of shape (None,)
    # Do not count the correct class itself (rank is counted from 0).
    if optimistic:
        return ranks
    return ranks - 1


def confidence(y_true: Any, y_pred: Any) -> Any:
    """Return the confidence of the prediction, that is the difference of the
    highest value and the second highest value. A prediction contributes to
    the overall score even if the prediction was incorrect.

    Standalone usage:
    >>> y_true = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.], [0.5, 0.5, 0.]]
    >>> c = confidence(y_true, y_pred)
    >>> assert c.shape == (3,)
    >>> c.numpy()
    array([0.1, 0.9, 0.0], dtype=float32)

    Args:
      y_true (batch of one-hot): One-hot ground truth values. Part of API,
        ignored in this function.
      y_pred (batch of probabilities): The prediction values.
    """
    del y_true  # unused

    # Take the first two predictions, top_two.values are their values,
    # top_two.indices are their indexes.
    top_two = keras.ops.top_k(y_pred, k=2)

    # Compute the difference of the top and the second prediction (regardless
    # if the prediction is correct or not).
    return top_two.values[:, 0] - top_two.values[:, 1]


@keras.utils.register_keras_serializable(package="SCAAML")
class MeanConfidence(MeanMetricWrapper):  # type: ignore[no-any-unimported,misc]
    """Calculates the average confidence of a prediction, that is the difference
    of the two largest values (regardless if the prediction is correct or not).

    Args:
      name (str): String name of the metric instance. Defaults to
        "mean_confidence".
      dtype (Optional): Data type of the metric result. Defaults to None.

    Standalone usage:

    >>> m = MeanConfidence()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    >>> m.result().numpy()
    0.4

    Usage with `compile()` API:

    ```python
    model.compile(optimizer="sgd",
                  loss="mse",
                  metrics=[MeanConfidence()])
    ```
    """

    def __init__(self,
                 name: str = "mean_confidence",
                 dtype: Optional[np.generic] = None) -> None:
        super().__init__(confidence, name, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Dictionary representation of this layer."""
        result = super().get_config()
        if "fn" in result:
            del result["fn"]
        return result  # type: ignore[no-any-return]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MeanConfidence":
        """Deserialize from a config."""
        return cls(**config)


@keras.utils.register_keras_serializable(package="SCAAML")
class MeanRank(MeanMetricWrapper):  # type: ignore[no-any-unimported,misc]
    """Calculates the mean rank of the correct class.

    The rank is the index of the correct class in the ordered predictions (the
    correct class is assumed to be the last one when a tie occurs). The rank is
    counted starting with zero for the correct prediction.

    Args:
      name (Optional): String name of the metric instance.
      dtype (Optional): Data type of the metric result.

    Standalone usage:

    >>> m = MeanRank()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer="sgd",
                  loss="mse",
                  metrics=[MeanRank()])
    ```
    """

    def __init__(self,
                 name: str = "mean_rank",
                 dtype: Optional[np.generic] = None) -> None:
        super().__init__(rank, name, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Dictionary representation of this layer."""
        result = super().get_config()
        if "fn" in result:
            del result["fn"]
        return result  # type: ignore[no-any-return]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MeanRank":
        """Deserialize from a config."""
        return cls(**config)


@keras.utils.register_keras_serializable(package="SCAAML")
class MaxRank(Metric):  # type: ignore[no-any-unimported,misc]
    """Calculates the maximum rank of the correct class.

    The rank is the index of the correct class in the ordered predictions (the
    correct class is assumed to be the last one when a tie occurs). The rank is
    counted starting with zero for the correct prediction.

    Args:
      name: (Optional) String name of the metric instance.
      dtype: (Optional) Data type of the metric result.

    Standalone usage:

    >>> m = MaxRank()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer="sgd",
                  loss="mse",
                  metrics=[MaxRank()])
    ```
    """

    def __init__(self, name: str = "max_rank", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.max_rank = self.add_weight(name="max_rank", initializer="zeros")

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
        rank_update = rank(y_true=y_true, y_pred=y_pred)
        rank_update = keras.ops.max(rank_update)
        self.max_rank.assign(keras.ops.maximum(self.max_rank, rank_update))

    def result(self) -> Any:
        """Return the result."""
        return keras.ops.cast(self.max_rank, dtype="int32")

    def reset_state(self) -> None:
        """Reset the state for new measurement."""
        # The state of the metric will be reset at the start of each epoch.
        self.max_rank.assign(0.0)


@keras.utils.register_keras_serializable(package="SCAAML")
class SignificanceTest(Metric):  # type: ignore[no-any-unimported,misc]
    """Calculates the probability that a random guess would get the same
    accuracy. Probability is in the interval [0, 1] (impossible to always). By
    convention one rejects the null hypothesis at a given p-value (say 0.005 if
    we want to be sure).

    Args:
      name: (Optional) String name of the metric instance.
      dtype: (Optional) Data type of the metric result.

    Standalone usage:

    >>> m = SignificanceTest()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.6, 0.4]])
    >>> m.result().numpy()
    0.25

    Usage with `compile()` API:

    ```python
    model.compile(optimizer="sgd",
                  loss="mse",
                  metrics=[SignificanceTest()])
    ```
    """

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
        return 1 - scipy.stats.binom.cdf(k - 1, n, 1 / possibilities)

    def reset_state(self) -> None:
        """Reset the state for new measurement."""
        # The state of the metric will be reset at the start of each epoch.
        self.seen.assign(0.0)
        self.correct.assign(0.0)
        self.possibilities.assign(0.0)
