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

import keras
import numpy as np

from scaaml.metrics import MeanConfidence
from scaaml.metrics import SignificanceTest
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
    m = SignificanceTest()
    m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.6, 0.4]])
    assert np.isclose(m.result(), 0.25)


def test_h0_significant():
    N = 10
    m = SignificanceTest()
    m.update_state([[0., 1.] for _ in range(N)], [[0.1, 0.9] for _ in range(N)])
    assert np.isclose(m.result(), 2**(-N))


def test_h0_all_wrong():
    m = SignificanceTest()
    m.update_state([[0., 1.], [1., 0.]], [[0.9, 0.1], [0.4, 0.6]])
    assert np.isclose(m.result(), 1.)


def test_h0_one_wrong():
    m = SignificanceTest()
    m.update_state([[0., 1.], [1., 0.]], [[0.1, 0.9], [0.4, 0.6]])
    assert np.isclose(m.result(), 0.75)


def get_mnist_model():
    """Return a CNN model.
    """
    input_shape = (28, 28)
    num_classes = 10

    input_data = keras.Input(shape=input_shape, name="input")

    x = input_data
    x = keras.layers.Reshape((*input_shape, 1))(x)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation="softmax", name="digit")(x)

    model = keras.Model(inputs=input_data, outputs=x)

    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            MeanConfidence(),
            MeanRank(),
            MaxRank(),
            #SignificanceTest(),  # need to get around np.array needed for scipy
        ],
    )
    return model


def test_load(tmp_path):
    """Test that a model can be loaded afterwards.
    """
    model = get_mnist_model()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 256
    x_test = x_test.astype("float32") / 256
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    # good accuracy is 98% or 99%:
    assert score[1] > 0.9

    # save and reaload the model
    model.save(tmp_path / "model.keras")
    model_loaded = keras.models.load_model(tmp_path / "model.keras")
    score = model_loaded.evaluate(x_test, y_test, verbose=0)
    # good accuracy is 98% or 99%:
    assert score[1] > 0.9
