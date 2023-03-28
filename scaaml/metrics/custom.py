# Copyright 2022 Google LLC
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
  tf.keras.metrics.TopKCategoricalAccuracy: How often the correct class
    is in the top K predictions (how often is rank less than K).
"""
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


@tf.function
def rank(y_true, y_pred, optimistic: bool = False):
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
    mul = tf.math.multiply(y_true, y_pred)
    # mul is of shape (None, N)

    # We get the predicted probability for the correct class (keeping the
    # dimension, so we can compare with y_pred).
    predicted_pr = tf.math.reduce_max(mul, axis=-1, keepdims=True)
    # predicted_pr is of shape (None, 1)

    # Boolean array where y_pred >= predicted_pr (True values denote the
    # classes that are assigned at least as high probability as the correct
    # class). If we are optimistic we count only y_pred > predicted_pr (break
    # ties in favor of the target).
    if optimistic:
        preferred = tf.math.greater(y_pred, predicted_pr)
    else:
        preferred = tf.math.greater_equal(y_pred, predicted_pr)
    # preferred is of shape (None, N)

    # A scalar, summing a 1. for each True and 0. for each False.
    ranks = tf.reduce_sum(tf.cast(preferred, "float32"), axis=-1)
    # ranks is of shape (None,)
    # Do not count the correct class itself (rank is counted from 0).
    if optimistic:
        return ranks
    else:
        return ranks - 1


@tf.function
def confidence(y_true, y_pred):
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
    top_two = tf.math.top_k(y_pred, k=2)

    # Compute the difference of the top and the second prediction (regardless
    # if the prediction is correct or not).
    return top_two.values[:, 0] - top_two.values[:, 1]


@tf.keras.utils.register_keras_serializable(package="SCAAML")
class MeanConfidence(keras.metrics.MeanMetricWrapper):
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

    def __init__(self, name: str = "mean_confidence", dtype=None) -> None:
        super().__init__(confidence, name, dtype=dtype)


@tf.keras.utils.register_keras_serializable(package="SCAAML")
class MeanRank(keras.metrics.MeanMetricWrapper):
    """Calculates the mean rank of the correct class.

    The rank is the index of the correct class in the ordered predictions (the
    correct class is assumed to be the last one when a tie occurs). The rank is
    counted starting with zero for the correct prediction.

    Args:
      name (Optional): String name of the metric instance.
      dtype (Optional): Data type of the metric result.
      decimals (Optional[int]): How many decimals to show. If None no
        rounding. Behaves as in np.round. Defaults to None.

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
                 dtype=None,
                 decimals: Optional[int] = None):
        super().__init__(rank, name, dtype=dtype)
        self._decimals = decimals

    def result(self):
        """Return the result, possibly rounded to the right number of digits.
        See the decimals parameter of the constructor.
        """
        # Get the result.
        res = super().result()

        # Check if rounding is necessary.
        if self._decimals is None:
            return res

        # Get numpy scalar and round.
        rounded = np.round(res.numpy(), self._decimals)
        # Cast back to tensor.
        return tf.convert_to_tensor(rounded, dtype=res.dtype)


@tf.keras.utils.register_keras_serializable(package="SCAAML")
class MaxRank(keras.metrics.Metric):
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

    def __init__(self, name: str = "max_rank", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_rank = self.add_weight(name="max_rank", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state.

        Args:
          y_true (batch of one-hot): One-hot ground truth values.
          y_pred (batch of one-hot): The prediction values.
          sample_weight (Optional weights): Does not make sense, as we count
            maximum.
        """
        del sample_weight  # unused
        rank_update = rank(y_true=y_true, y_pred=y_pred)
        rank_update = tf.math.reduce_max(rank_update)
        self.max_rank.assign(tf.math.maximum(self.max_rank, rank_update))

    def result(self):
        """Return the result."""
        return tf.cast(self.max_rank, dtype=tf.int32)

    def reset_state(self):
        """Reset the state for new measurement."""
        # The state of the metric will be reset at the start of each epoch.
        self.max_rank.assign(0.0)
