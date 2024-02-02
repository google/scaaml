# Copyright 2020-2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configure tests."""

import os
from pathlib import Path
import pytest

from tensorflow.python.autograph.core import config
import tensorflow as tf

TEST_ROOT_DIR = Path(__file__).parent
TEST_DATA_ROOT = TEST_ROOT_DIR / "data"


@pytest.fixture(scope="session")
def scald_shards_path():
    return [
        str(TEST_DATA_ROOT / "scald/tinyaes.npz"),
        str(TEST_DATA_ROOT / "scald/mbed.npz"),
    ]


@pytest.fixture(autouse=True)
def disable_autograph_in_coverage() -> None:
    """TensorFlow is doing magic with Python AST that excludes functions such
    as scaaml.metrics.custom.rank from being marked as covered by unittests.
    Modifiable by the environment variable `DISABLE_AUTOGRAPH`.

    https://github.com/tensorflow/tensorflow/issues/33759
    """
    if not os.getenv("DISABLE_AUTOGRAPH"):
        return  # pragma: no cover
    config.CONVERSION_RULES = (
        config.DoNotConvert("scaaml"),) + config.CONVERSION_RULES
    tf.config.run_functions_eagerly(True)
