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

import os
from unittest.mock import MagicMock, patch

import numpy as np

from scaaml.io import Dataset
from scaaml.stats import ExampleIterator


@patch.object(Dataset, "from_config")
def test_init_default(mock_from_config):
    all_shards = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock()
    ]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    example_iterator = ExampleIterator(dataset_path=ds_path)

    assert example_iterator._shard_idx == 0
    assert example_iterator._shards_list == [
        (0, all_shards[0], Dataset.TRAIN_SPLIT),
        (1, all_shards[1], Dataset.TRAIN_SPLIT),
        (0, all_shards[2], Dataset.TEST_SPLIT),
        (0, all_shards[3], Dataset.HOLDOUT_SPLIT),
        (1, all_shards[4], Dataset.HOLDOUT_SPLIT),
    ]
    mock_from_config.assert_called_once_with(dataset_path=ds_path,
                                             verbose=False)


@patch.object(Dataset, "from_config")
def test_init_single_split(mock_from_config):
    all_shards = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock()
    ]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    example_iterator = ExampleIterator(dataset_path=ds_path,
                                       split=Dataset.HOLDOUT_SPLIT)

    assert example_iterator._shard_idx == 0
    assert example_iterator._shards_list == [
        (0, all_shards[3], Dataset.HOLDOUT_SPLIT),
        (1, all_shards[4], Dataset.HOLDOUT_SPLIT),
    ]


@patch.object(Dataset, "from_config")
def test_init_single_group(mock_from_config):
    all_shards = [
        {
            "mock": MagicMock(),
            "group": 0
        },
        {
            "mock": MagicMock(),
            "group": 1
        },
        {
            "mock": MagicMock(),
            "group": 2
        },
        {
            "mock": MagicMock(),
            "group": 0
        },
        {
            "mock": MagicMock(),
            "group": 1
        },
        {
            "mock": MagicMock(),
            "group": 0
        },
    ]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    example_iterator = ExampleIterator(dataset_path=ds_path, group=1)

    assert example_iterator._shard_idx == 0
    assert example_iterator._shards_list == [
        (1, all_shards[1], Dataset.TRAIN_SPLIT),
        (1, all_shards[4], Dataset.HOLDOUT_SPLIT),
    ]


@patch.object(Dataset, "from_config")
def test_init_single_part(mock_from_config):
    all_shards = [
        {
            "mock": MagicMock(),
            "part": 0
        },
        {
            "mock": MagicMock(),
            "part": 1
        },
        {
            "mock": MagicMock(),
            "part": 2
        },
        {
            "mock": MagicMock(),
            "part": 0
        },
        {
            "mock": MagicMock(),
            "part": 1
        },
        {
            "mock": MagicMock(),
            "part": 0
        },
    ]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    example_iterator = ExampleIterator(dataset_path=ds_path, part=2)

    assert example_iterator._shard_idx == 0
    assert example_iterator._shards_list == [
        (0, all_shards[2], Dataset.TEST_SPLIT),
    ]


@patch.object(Dataset, "from_config")
def test_len(mock_from_config):
    examples_per_shard = 17
    all_shards = [
        {
            "mock": MagicMock(),
            "examples": examples_per_shard
        },
        {
            "mock": MagicMock(),
            "examples": examples_per_shard
        },
        {
            "mock": MagicMock(),
            "examples": examples_per_shard
        },
        {
            "mock": MagicMock(),
            "examples": examples_per_shard
        },
        {
            "mock": MagicMock(),
            "examples": examples_per_shard
        },
    ]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    example_iterator = ExampleIterator(dataset_path=ds_path)

    assert len(example_iterator) == len(all_shards) * examples_per_shard


@patch.object(Dataset, "inspect")
@patch.object(Dataset, "from_config")
def test_iteration(mock_from_config, mock_inspect):
    all_shards = [
        {
            "mock": MagicMock(),
            "inspect": MagicMock(),
            "examples": 10
        },
        {
            "mock": MagicMock(),
            "inspect": MagicMock(),
            "examples": 20
        },
        {
            "mock": MagicMock(),
            "inspect": MagicMock(),
            "examples": 30
        },
        {
            "mock": MagicMock(),
            "inspect": MagicMock(),
            "examples": 30
        },
        {
            "mock": MagicMock(),
            "inspect": MagicMock(),
            "examples": 10
        },
    ]
    all_examples = np.random.random(100)
    all_shards[0]["inspect"] = all_examples[:10]
    all_shards[1]["inspect"] = all_examples[10:30]
    all_shards[2]["inspect"] = all_examples[30:60]
    all_shards[3]["inspect"] = all_examples[60:90]
    all_shards[4]["inspect"] = all_examples[90:]
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [all_shards[0], all_shards[1]],
        Dataset.TEST_SPLIT: [all_shards[2]],
        Dataset.HOLDOUT_SPLIT: [all_shards[3], all_shards[4]],
    }
    ds_path = "/mnt/not_a_dataset"

    def mock_inspect_side_effect(dataset_path, split, shard_id, num_example,
                                 verbose):
        assert dataset_path == ds_path
        assert verbose == False
        shard = mock_from_config.return_value.shards_list[split][shard_id]
        assert num_example == shard["examples"]
        result = MagicMock()
        result.as_numpy_iterator.return_value = iter(shard["inspect"])
        return result

    mock_inspect.side_effect = mock_inspect_side_effect

    example_iterator = ExampleIterator(dataset_path=ds_path)

    for i, example in enumerate(example_iterator):
        assert example == all_examples[i]
