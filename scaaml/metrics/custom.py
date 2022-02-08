"""Custom metrics: MeanRank, MaxRank.

Related metrics:
  MinRank: Is zero as soon as accuracy is non-zero.
  tf.keras.metrics.TopKCategoricalAccuracy: How often the correct class
    is in the top K predictions (how often is rank less than K).
"""
import tensorflow as tf
from tensorflow import keras


@tf.function
def rank(y_true, y_pred):
    """Calculates the rank of the correct class (counted from 0). If the
    prediction is correct, the rank is equal to zero. If the correct class is
    the least probable according to y_pred, than the rank is equal to number
    of classes - 1.

    When there is a tie in y_pred (two or more classes are assigned the same
    probability) then the correct class is assumed to be the least probable.

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
      y_true: One-hot ground truth values.
      y_pred: The prediction values.

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
    # class).
    preferred = tf.math.greater_equal(y_pred, predicted_pr)
    # preferred is of shape (None, N)

    # A scalar, summing a 1. for each True and 0. for each False.
    ranks = tf.reduce_sum(tf.cast(preferred, 'float32'), axis=-1)
    # ranks is of shape (None,)
    # Do not count the correct class itself (rank is counted from 0).
    return ranks - 1


@tf.keras.utils.register_keras_serializable(package="SCAAML")
class MeanRank(keras.metrics.MeanMetricWrapper):
    """Calculates the mean rank of the correct class.

    The rank is the index of the correct class in the ordered predictions (the
    correct class is assumed to be the last one when a tie occurs). The rank is
    counted starting with zero for the correct prediction.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = MeanRank()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[MeanRank()])
    ```
    """
    def __init__(self, name='mean_rank', dtype=None):
        super().__init__(rank, name, dtype=dtype)


@tf.keras.utils.register_keras_serializable(package="SCAAML")
class MaxRank(keras.metrics.Metric):
    """Calculates the maximum rank of the correct class.

    The rank is the index of the correct class in the ordered predictions (the
    correct class is assumed to be the last one when a tie occurs). The rank is
    counted starting with zero for the correct prediction.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = MaxRank()
    >>> m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[MaxRank()])
    ```
    """
    def __init__(self, name: str = "max_rank", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_rank = self.add_weight(name="max_rank", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state.

        Args:
          y_true: One-hot ground truth values.
          y_pred: The prediction values.
          sample_weight: Does not make sense, as we count maximum.
        """
        rank_update = rank(y_true=y_true, y_pred=y_pred)
        rank_update = tf.math.reduce_max(rank_update)
        self.max_rank.assign(tf.math.maximum(self.max_rank, rank_update))

    def result(self):
        """Return the result."""
        return self.max_rank

    def reset_state(self):
        """Reset the state for new measurement."""
        # The state of the metric will be reset at the start of each epoch.
        self.max_rank.assign(0.0)
