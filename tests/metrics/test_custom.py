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

from scaaml.metrics import MaxRank, MeanRank
from scaaml.metrics.custom import rank


def rank_slow_1d(y_true, y_pred) -> float:
    """Slow, but correct implementation for single prediction."""
    assert len(y_true) == len(y_pred)
    correct_class = 0
    for i, value in enumerate(y_true):
        if value == 1:
            correct_class = i
    result = 0
    for pred in y_pred:
        if pred >= y_pred[correct_class]:
            result += 1
    return float(result - 1)


def rank_slow(y_true, y_pred):
    """Slow, but correct implementation for a batch of predictions."""
    return np.array(
        [rank_slow_1d(y_true[i], y_pred[i]) for i in range(len(y_true))])


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


def test_meanrank_doc():
    r = MeanRank()
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 0.5


def test_meanrank_decimals_no_rounding():
    # No rounding
    r = MeanRank(decimals=None)
    r.update_state([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                   [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    assert r.result().numpy() == 0.25


def test_meanrank_1_decimals():
    # One decimal
    r = MeanRank(decimals=1)
    r.update_state([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                   [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    assert np.isclose(r.result().numpy(), 0.2)


def test_maxrank_doc():
    r = MaxRank()
    r.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.5, 0.5]])
    assert r.result().numpy() == 1
    assert r.result().numpy().dtype == np.int32
