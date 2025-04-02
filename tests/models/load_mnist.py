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

import sys

import keras

# Needed to load the model.
from scaaml.models.gpam import *


def main():
    model = keras.models.load_model(sys.argv[1])
    model.summary()

    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28 * 28)
    y_test = keras.utils.to_categorical(y_test, 10)

    score = model.evaluate(x_test, y_test)
    print("[load_and_test loaded] Test loss:", score[0])
    print("[load_and_test loaded] Test accuracy:", score[1])
    assert score[1] > 0.97


if __name__ == "__main__":
    main()
