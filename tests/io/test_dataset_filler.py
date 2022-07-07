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

from unittest.mock import MagicMock, patch
from typing import Dict, List

import pytest

from scaaml.io import Dataset
from scaaml.io import DatasetFiller
from scaaml.io.dataset_filler import _DatasetFillerContext


def test_context_manager():
    """Test with."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = 64

    with DatasetFiller(
            dataset=mock_dataset,
            plaintexts_per_key=256,
            repetitions=1,
    ) as dataset_filler:
        # Ensure that the context type is right.
        assert type(dataset_filler) is _DatasetFillerContext


def test_context_manager_close_shard():
    """Assert that we do not call close_shard at __exit__ if there has been
    no example."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = 64

    with DatasetFiller(
            dataset=mock_dataset,
            plaintexts_per_key=256,
            repetitions=1,
    ) as dataset_filler:
        pass

    mock_dataset.close_shard.assert_not_called()


def test_skip_examples():
    """Test that skip examples calls _update_counters that many times."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = 7
    # skip 6 shards
    skip_examples_n = 6 * mock_dataset.examples_per_shard

    with DatasetFiller(
            dataset=mock_dataset,
            plaintexts_per_key=256,
            repetitions=1,
            skip_examples=skip_examples_n,
    ) as dataset_filler:
        assert dataset_filler._written_examples == skip_examples_n
        assert dataset_filler.written_examples_not_skipped == 0


def test_skip_examples_skip_shards_only():
    """Test that skip examples always skips whole shards."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = 7
    skip_examples_n = 13
    # Testing skipping not a whole number of shards.
    assert mock_dataset.examples_per_shard % skip_examples_n

    with pytest.raises(ValueError) as value_error:
        with DatasetFiller(
                dataset=mock_dataset,
                plaintexts_per_key=256,
                repetitions=1,
                skip_examples=skip_examples_n,
        ) as dataset_filler:
            pass
        msg = f'{skip_examples_n} is not divisible by {mock_dataset.examples_per_shard}'
        assert msg == str(value_error.value)


def test_add_examples():
    """Test that skip examples always skips whole shards."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = 7
    shards = 100

    examples = [(MagicMock(), MagicMock())
                for _ in range(shards * mock_dataset.examples_per_shard)]

    with DatasetFiller(
            dataset=mock_dataset,
            plaintexts_per_key=5,
            repetitions=1,
    ) as dataset_filler:
        for attack_points, measurement in examples:
            dataset_filler.write_example(
                attack_points=attack_points,
                measurement=measurement,
                current_key=[1, 2, 3],
                split_name=Dataset.TEST_SPLIT,
                chip_id=123,
            )

    # Every shard has been opened.
    mock_dataset.new_shard.call_count == shards
    # The last shard has been closed.
    mock_dataset.close_shard.call_count == 1
    # All examples have been written.
    assert mock_dataset.write_example.call_args_list == [({
        'attack_points': attack_points,
        'measurement': measurement,
    },) for attack_points, measurement in examples]


def get_mock_dataset_method_calls(examples_per_shard, skip_shards,
                                  write_example_kwargs: List[Dict],
                                  plaintexts_per_key, repetitions):
    """Get all method calls called on dataset."""
    mock_dataset = MagicMock()
    mock_dataset.examples_per_shard = examples_per_shard
    skip_n_examples = skip_shards * examples_per_shard

    with DatasetFiller(
            dataset=mock_dataset,
            plaintexts_per_key=plaintexts_per_key,
            repetitions=repetitions,
            skip_examples=skip_n_examples,
    ) as dataset_filler:
        for kwargs in write_example_kwargs[skip_n_examples:]:
            dataset_filler.write_example(**kwargs)

    return mock_dataset.method_calls


def test_few_examples():
    examples = [{
        'attack_points': MagicMock(),
        'measurement': MagicMock(),
    } for _ in range(8)]
    write_example_kwargs = [{
        'attack_points': example['attack_points'],
        'measurement': example['measurement'],
        'current_key': [1, 2, 3],
        'split_name': Dataset.TEST_SPLIT,
        'chip_id': 123,
    } for example in examples]

    method_calls = get_mock_dataset_method_calls(
        examples_per_shard=2,
        skip_shards=0,
        write_example_kwargs=write_example_kwargs,
        plaintexts_per_key=4,
        repetitions=1,
    )

    expected_method_calls = [
        ('new_shard', (), {
            'key': [1, 2, 3],
            'part': 0,
            'group': 0,
            'split': Dataset.TEST_SPLIT,
            'chip_id': 123,
        }),
        ('write_example', examples[0]),
        ('write_example', examples[1]),
        ('new_shard', (), {
            'key': [1, 2, 3],
            'part': 1,
            'group': 0,
            'split': Dataset.TEST_SPLIT,
            'chip_id': 123,
        }),
        ('write_example', examples[2]),
        ('write_example', examples[3]),
        ('new_shard', (), {
            'key': [1, 2, 3],
            'part': 0,
            'group': 0,
            'split': Dataset.TEST_SPLIT,
            'chip_id': 123,
        }),
        ('write_example', examples[4]),
        ('write_example', examples[5]),
        ('new_shard', (), {
            'key': [1, 2, 3],
            'part': 1,
            'group': 0,
            'split': Dataset.TEST_SPLIT,
            'chip_id': 123,
        }),
        ('write_example', examples[6]),
        ('write_example', examples[7]),
        ('close_shard',),
    ]
    assert method_calls == expected_method_calls


def test_same_calls():
    """Test that when we skip some shards we do the same calls (same part and
    group number)."""
    examples = [{
        'attack_points': MagicMock(),
        'measurement': MagicMock(),
    } for _ in range(256)]
    write_example_kwargs = [{
        'attack_points': example['attack_points'],
        'measurement': example['measurement'],
        'current_key': [1, 2, 3],
        'split_name': Dataset.TEST_SPLIT,
        'chip_id': 123,
    } for example in examples]
    examples_per_shard: int = 4
    plaintexts_per_key: int = 8
    repetitions: int = 1

    method_calls = get_mock_dataset_method_calls(
        examples_per_shard=examples_per_shard,
        skip_shards=0,
        write_example_kwargs=write_example_kwargs,
        plaintexts_per_key=plaintexts_per_key,
        repetitions=repetitions,
    )

    for skip_shards in range(len(examples) // examples_per_shard):
        truncated_method_calls = get_mock_dataset_method_calls(
            examples_per_shard=examples_per_shard,
            skip_shards=skip_shards,
            write_example_kwargs=write_example_kwargs,
            plaintexts_per_key=plaintexts_per_key,
            repetitions=repetitions,
        )
        skip_calls = skip_shards * (1 + examples_per_shard)
        # There is the right number of method calls
        assert len(truncated_method_calls) + skip_calls == len(method_calls)
        # The method calls have the right parameters
        assert truncated_method_calls == method_calls[skip_calls:]
