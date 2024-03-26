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

import numpy as np

from scaaml.metrics import MeanConfidence
from scaaml.metrics import H0
from scaaml.metrics import MaxRank, MeanRank
from scaaml.metrics.custom import confidence
from scaaml.metrics.custom import rank


def rank_slow_1d(y_true, y_pred, optimistic) -> float:
    """Slow, but correct implementation for single prediction."""
    assert len(y_true) == len(y_pred)
    correct_class = 0
    for i, value in enumerate(y_true):
        if value == 1:
            correct_class = i
    result = 0
    for pred in y_pred:
        if optimistic:
            # Break ties in favor of target
            if pred > y_pred[correct_class]:
                result += 1
        else:
            # Break ties in favor of other classes
            if pred >= y_pred[correct_class]:
                result += 1
    # When all probabilities are different we should return the same result
    # (a number between 0 and #classes - 1).
    if optimistic:
        return float(result)
    else:
        return float(result - 1)


def rank_slow(y_true, y_pred, optimistic=False):
    """Slow, but correct implementation for a batch of predictions."""
    return np.array([
        rank_slow_1d(y_true[i], y_pred[i], optimistic)
        for i in range(len(y_true))
    ])


def test_rank_random_ties_optimistic():
    # Make the test deterministic.
    np.random.seed(42)
    byte_values = 256
    batch_size = 1_000

    def r_y_true():
        y_true = np.zeros(byte_values)
        y_true[np.random.randint(byte_values)] = 1.
        return y_true

    y_true = [r_y_true() for _ in range(batch_size)]
    y_pred = [
        np.around(np.random.random(byte_values), 1) for _ in range(batch_size)
    ]

    r = rank(y_true, y_pred, optimistic=True)

    assert r.shape == (batch_size,)
    assert (r.numpy() == rank_slow(y_true, y_pred, optimistic=True)).all()


def test_rank_random_ties():
    # Make the test deterministic.
    np.random.seed(42)
    byte_values = 256
    batch_size = 1_000

    def r_y_true():
        y_true = np.zeros(byte_values)
        y_true[np.random.randint(byte_values)] = 1.
        return y_true

    y_true = [r_y_true() for _ in range(batch_size)]
    y_pred = [
        np.around(np.random.random(byte_values), 1) for _ in range(batch_size)
    ]

    r = rank(y_true, y_pred)

    assert r.shape == (batch_size,)
    assert (r.numpy() == rank_slow(y_true, y_pred)).all()


def test_rank_random():
    # Make the test deterministic.
    np.random.seed(42)
    byte_values = 256
    batch_size = 1_000

    def r_y_true():
        y_true = np.zeros(byte_values)
        y_true[np.random.randint(byte_values)] = 1.
        return y_true

    y_true = [r_y_true() for _ in range(batch_size)]
    y_pred = [np.random.random(byte_values) for _ in range(batch_size)]

    r = rank(y_true, y_pred)

    assert r.shape == (batch_size,)
    assert (r.numpy() == rank_slow(y_true, y_pred)).all()


def test_rank_correct_pred():
    matrix_side = 6
    y_true = np.eye(matrix_side)
    r = rank(y_true, y_true)
    assert r.shape == (matrix_side,)
    assert (r.numpy() == rank_slow(y_true, y_true)).all()


def test_rank_doc():
    y_true = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
    y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.], [0.5, 0.5, 0.]]

    r = rank(y_true, y_pred)

    assert r.shape == (3,)
    assert (r.numpy() == np.array([1., 0., 1.], dtype=np.float32)).all()
    assert (r.numpy() == rank_slow(y_true, y_pred)).all()


def test_mean_rank_doc():
    r = MeanRank()
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 0.5


def test_mean_rank_decimals_no_rounding():
    # No rounding
    r = MeanRank(decimals=None)
    r.update_state([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                   [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    assert r.result().numpy() == 0.25


def test_mean_rank_1_decimals():
    # One decimal
    r = MeanRank(decimals=1)
    r.update_state([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                   [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    assert np.isclose(r.result().numpy(), 0.2)


def test_max_rank_doc():
    r = MaxRank()
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 1
    assert r.result().numpy().dtype == np.int32


def test_max_rank_reset():
    r = MaxRank()
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 1
    assert r.result().numpy().dtype == np.int32

    r.reset_state()
    assert r.result().numpy() == 0
    assert r.result().numpy().dtype == np.int32
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 1
    assert r.result().numpy().dtype == np.int32


def test_confidence_doc():
    y_true = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
    y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.], [0.5, 0.5, 0.]]

    c = confidence(y_true, y_pred)

    assert c.shape == (3,)
    assert np.allclose(c.numpy(), np.array([0.1, 0.9, 0.0], dtype=np.float32))


def test_mean_confidence_doc():
    m = MeanConfidence()
    m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert np.isclose(m.result().numpy(), 0.4)


def test_confidence_random():
    # Make the test deterministic
    np.random.seed(42)
    byte_values = 256
    batch_size = 1_000

    # Targets do not matter
    y_true = [np.random.random(byte_values) for _ in range(batch_size)]
    y_pred = [np.random.random(byte_values) for _ in range(batch_size)]

    c = confidence(y_true, y_pred)

    # Compute expected result
    s = np.sort(y_pred)
    expected = s[:, -1] - s[:, -2]

    assert c.shape == (batch_size,)
    assert np.allclose(c.numpy(), expected)


def test_optimistic_with_all_different():
    # Make the test deterministic
    np.random.seed(42)
    byte_values = 256
    batch_size = 1_000

    y_pred = []
    y_true = []
    # Only different probabilities
    for _ in range(batch_size):
        predictions = np.random.random(byte_values)
        _, counts = np.unique(predictions, return_counts=True)
        if (counts == 1).all():
            # All are unique
            y_pred.append(predictions)
            target = np.zeros(byte_values)
            target[np.random.randint(0, byte_values)] = 1
            y_true.append(target)
    # We should have enough samples
    assert len(y_pred) >= batch_size / 2

    optimistic = rank(y_true, y_pred, optimistic=True)
    pesimistic = rank(y_true, y_pred, optimistic=False)

    # Get rid of Tensor to have .all method.
    optimistic = np.array(optimistic)
    pesimistic = np.array(pesimistic)
    assert (optimistic == pesimistic).all()


def test_h0_doc():
    m = H0()
    m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.6, 0.4]])
    assert np.isclose(m.result(), 0.25)


def test_h0_significant():
    N = 10
    m = H0()
    m.update_state([[0., 1.] for _ in range(N)], [[0.1, 0.9] for _ in range(N)])
    assert np.isclose(m.result(), 2**(-N))


def test_h0_all_wrong():
    m = H0()
    m.update_state([[0., 1.], [1., 0.]], [[0.9, 0.1], [0.4, 0.6]])
    assert np.isclose(m.result(), 1.)


def test_h0_one_wrong():
    m = H0()
    m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.4, 0.6]])
    assert np.isclose(m.result(), 0.75)
