# Copyright 2021 Google LLC
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

import os
import copy
import json
import math
from pathlib import Path
import pytest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

import scaaml
from scaaml.io import Dataset
from scaaml.io import DatasetFiller
from scaaml.io.shard import Shard
from scaaml.io import utils as siutils
from scaaml.io.errors import DatasetExistsError
from scaaml.io.reshape import reshape_into_new_dataset
from scaaml.stats import ExampleIterator


def save_dataset(root_dir, all_examples, measurements_info, attack_points_info,
                 examples_per_shard, dtype):
    ds = Dataset(
        root_path=str(root_dir),
        shortname="testing_ds",
        architecture="test",
        implementation="alpha_testing",
        algorithm="candidate",
        version=1,
        firmware_sha256="0xcoffee",
        description="just testing",
        examples_per_shard=examples_per_shard,
        measurements_info=measurements_info,
        attack_points_info=attack_points_info,
        url="https://test.test",
        firmware_url="https://firmware.test",
        measurement_dtype=tf.float32 if dtype == "float32" else tf.float16,
    )
    ds_path = ds.path

    with DatasetFiller(ds, 4, 1) as dataset_filler:
        for example in all_examples:
            dataset_filler.write_example(
                attack_points=example["attack_points"],
                measurement=example["measurement"],
                current_key=example["attack_points"]["ap_one"],
                split_name=Dataset.TRAIN_SPLIT,
                chip_id=42,
            )

    return ds_path


def dataset_e2e(dtype, root_dir):
    """Test saving a shard and loading the data from it."""
    examples_per_shard = 4
    measurements_info = {
        "trace1": {
            "type": "power",
            "len": 208,
        },
    }
    attack_points_info = {
        "ap_one": {
            "len": 16,
            "max_val": 42,  # number of classes
        },
        "ap_two": {
            "len": 16,
            "max_val": 150,  # number of classes
        },
    }

    rng = np.random.default_rng(13)  # Make tests deterministic
    all_examples = [{
        "measurement": {
            measurement_name:
                rng.standard_normal(
                    size=(measurements_info[measurement_name]["len"],)
                ).astype(np.float32 if dtype == "float32" else np.float16)
            for measurement_name in measurements_info
        },
        "attack_points": {
            attack_point_name:
                rng.integers(
                    low=0,
                    high=attack_points_info[attack_point_name]["max_val"],
                    size=(attack_points_info[attack_point_name]["len"],),
                )
            for attack_point_name in attack_points_info
        },
    }
                    for _ in range(10 * examples_per_shard)]

    ds_path = save_dataset(
        root_dir=root_dir,
        all_examples=all_examples,
        measurements_info=measurements_info,
        attack_points_info=attack_points_info,
        examples_per_shard=examples_per_shard,
        dtype=dtype,
    )

    for parsed, remembered in zip(
            ExampleIterator(ds_path),
            all_examples,
    ):
        for measurement_name in measurements_info:
            assert np.allclose(parsed[measurement_name],
                               remembered["measurement"][measurement_name])
        for attack_point_name in attack_points_info:
            assert np.allclose(parsed[attack_point_name],
                               remembered["attack_points"][attack_point_name])


def test_dataset_e2e_float32(tmp_path):
    root_dir = tmp_path / "testingdataset"
    root_dir.mkdir()

    dataset_e2e(dtype="float32", root_dir=root_dir)


def test_dataset_e2e_float16(tmp_path):
    root_dir = tmp_path / "testingdataset"
    root_dir.mkdir()

    dataset_e2e(dtype="float16", root_dir=root_dir)
