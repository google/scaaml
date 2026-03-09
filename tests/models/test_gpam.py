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

import numpy as np
import tensorflow as tf
import keras

from scaaml.models import get_gpam_model


def test_train_save_load(tmp_path):
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    save_path = str(tmp_path / "mnist.keras")

    outputs = {"label": {"max_val": 10}}
    model = get_gpam_model(
        inputs={"trace1": {
            "min": 0,
            "delta": 256
        }},
        outputs=outputs,
        output_relations=[],
        trace_len=28 * 28,
        merge_filter_1=28,
        merge_filter_2=14,
        patch_size=28,
    )
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adafactor(0.1),
        loss=["categorical_crossentropy" for _ in range(len(outputs))],
        metrics={name: ["acc"] for name in outputs},
    )
    model.summary()

    # Work with a subset of data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    n_train: int = 1_500
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]
    n_test: int = 200
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    x_train = x_train.reshape(-1, 28 * 28)
    y_train = keras.utils.to_categorical(y_train, 10)
    x_test = x_test.reshape(-1, 28 * 28)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Train only a little bit to ensure trainable weights were changed (as
    # opposed to initialized).
    _ = model.fit(
        x_train,
        y_train,
        batch_size=10,
        epochs=1,
    )

    model.save(save_path)

    min_accuracy: float = 0.17

    score = model.evaluate(x_test, y_test)
    print("[orig] Test loss:", score[0])
    print("[orig] Test accuracy:", score[1])
    assert score[1] > min_accuracy

    loaded_model = keras.models.load_model(save_path)
    loaded_model.summary()
    score = loaded_model.evaluate(x_test, y_test)
    print("[loaded] Test loss:", score[0])
    print("[loaded] Test accuracy:", score[1])
    assert score[1] > min_accuracy

    # Make sure the loaded model is the same layer by layer.
    def match(i, x):
        print(f"model.layers[{i}].name = {model.layers[i].name}")
        np.testing.assert_allclose(
            model.layers[i](x),
            loaded_model.layers[i](x),
        )

    match(1, np.random.uniform(size=(1, 28 * 28)))
    match(2, np.random.uniform(size=(1, 28, 28)))
    match(3, np.random.uniform(size=(1, 28, 28)))
    match(4, np.random.uniform(size=(1, 28, 192)))
    match(5, np.random.uniform(size=(1, 28, 192)))
    match(6, np.random.uniform(size=(1, 28, 192)))
    match(7, np.random.uniform(size=(1, 28, 192)))
    match(9, np.random.uniform(size=(1, 28, 576)))
    match(10, np.random.uniform(size=(1, 28, 576)))
    match(11, np.random.uniform(size=(1, 26, 28)))
    match(12, np.random.uniform(size=(1, 13, 28)))
    match(13, np.random.uniform(size=(1, 13, 28)))
    match(14, np.random.uniform(size=(1, 11, 14)))
    match(15, np.random.uniform(size=(1, 11, 14)))
    match(16, np.random.uniform(size=(1, 11)))
    match(17, np.random.uniform(size=(1, 11)))
    match(18, np.random.uniform(size=(1, 11)))
    match(19, np.random.uniform(size=(1, 10)))
    match(20, np.random.uniform(size=(1, 10)))
    match(21, np.random.uniform(size=(1, 10)))
    match(22, np.random.uniform(size=(1, 10)))
    match(23, np.random.uniform(size=(1, 10)))
