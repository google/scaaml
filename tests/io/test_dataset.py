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
from scaaml.io.shard import Shard
from scaaml.io import utils as siutils
from scaaml.io.errors import DatasetExistsError
from scaaml.io.reshape import reshape_into_new_dataset


def dataset_constructor_kwargs(root_path, **kwargs):
    """Return key-value arguments for dataset constructor. In order to add or
    change arguments pass arguments to this function. The parameter root_path
    must be always specified.

    Args:
      root_path: The root_path for the Dataset.
      kwargs: Arguments to update the defaults.

    Example use:
      # Set "version" to 2, add new parameter "new_par" to "new_value" and
      # remove "firmware_url":
      result = dataset_constructor_kwargs(root_path=root_path,
                                          version=2,
                                          new_par="new_value")
      del result["firmware_url"]
    """
    result = {
        "root_path": root_path,
        "shortname": "shortname",
        "architecture": "architecture",
        "implementation": "implementation",
        "algorithm": "algorithm",
        "version": 1,
        "paper_url": "http://paper.url",
        "firmware_url": "http://firmware.url",
        "licence": "CC BY 4.0",
        "description": "description",
        "url": "http://download.url",
        "firmware_sha256": "abc123",
        "examples_per_shard": 1,
        "measurements_info": {
            "trace1": {
                "type": "power",
                "len": 1024,
            }
        },
        "attack_points_info": {
            "key": {
                "len": 16,
                "max_val": 256
            },
        }
    }
    result.update(kwargs)
    return result


def test_defaultdict(tmp_path):
    np.random.seed(42)
    # Create the dataset, write some examples, reload and write some more. If
    # reload does not create a defaultdict, we get a KeyError (for "test"
    # split).
    chip_id = 13
    kwargs = dataset_constructor_kwargs(root_path=tmp_path)
    ds = Dataset.get_dataset(**kwargs)
    key = np.random.randint(0, 255, 16)
    trace1 = np.random.random(1024)
    trace2 = np.random.random(1024)
    ds.new_shard(key=key,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()
    del ds
    key2 = np.random.randint(0, 255, 16)
    ds = Dataset.get_dataset(**kwargs)
    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TEST_SPLIT,
                 group=1,
                 chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace2})
    ds.close_shard()


def test_mutable_defaults(tmp_path):
    """If min_values defaults to {} this test fails (default value will get
    changed -- see mutable default arguments)."""
    np.random.seed(24)
    path_a = tmp_path / "a"
    path_a.mkdir(parents=True, exist_ok=True)
    ds1 = Dataset.get_dataset(**dataset_constructor_kwargs(root_path=path_a))
    key = np.random.randint(0, 255, 16)
    trace1 = np.zeros(1024, dtype=np.float64)
    trace1[2] = 0.8  # max_val is written before deleting ds
    trace1[1] = -0.3  # max_val is written before deleting ds
    chip_id = 1
    ds1.new_shard(key=key,
                  part=0,
                  split=Dataset.TRAIN_SPLIT,
                  group=0,
                  chip_id=chip_id)
    ds1.write_example({"key": key}, {"trace1": trace1})
    ds1.close_shard()

    # Different dataset
    path_b = tmp_path / "b"
    path_b.mkdir(parents=True, exist_ok=True)
    ds2 = Dataset.get_dataset(**dataset_constructor_kwargs(root_path=path_b))
    assert ds2.min_values == {"trace1": math.inf}
    assert ds2.max_values == {"trace1": 0.}


def test_version_old_software(tmp_path):
    """Newer version of scaaml was used to capture the dataset."""
    # Create the dataset
    scaaml.__version__ = "2.0.0"
    ds = Dataset.get_dataset(**dataset_constructor_kwargs(root_path=tmp_path))
    scaaml.__version__ = "1.2.3"
    # Reload the dataset raises
    with pytest.raises(ValueError) as value_error:
        ds = Dataset.from_config(ds.path)
    assert "SCAAML module is outdated" in str(value_error.value)


def test_version_newer_software(tmp_path):
    """Older version of scaaml was used to capture the dataset."""
    # Create the dataset
    scaaml.__version__ = "1.2.3"
    ds = Dataset.get_dataset(**dataset_constructor_kwargs(root_path=tmp_path))
    # Increment library version
    scaaml.__version__ = "1.3.3"
    # Reload the dataset
    ds = Dataset.from_config(ds.path)


def test_version_same(tmp_path):
    """Same version of scaaml was used to capture the dataset."""
    # Create the dataset
    ds = Dataset.get_dataset(**dataset_constructor_kwargs(root_path=tmp_path))
    # Reload the dataset
    ds = Dataset.from_config(ds.path)


def test_firmware_url_mandatory(tmp_path):
    kwargs = dataset_constructor_kwargs(root_path=tmp_path, firmware_url="")
    with pytest.raises(ValueError) as value_error:
        ds = Dataset.get_dataset(**kwargs)
    assert "Firmware URL is required" in str(value_error.value)


def test_get_config_dictionary_urls(tmp_path):
    licence = "some licence"
    firmware_url = "firmware url"
    paper_url = "paper URL"
    url = "U R L"
    kwargs = dataset_constructor_kwargs(root_path=tmp_path,
                                        licence=licence,
                                        firmware_url=firmware_url,
                                        paper_url=paper_url,
                                        url=url)
    # Create the dataset
    ds = Dataset.get_dataset(**kwargs)
    conf = ds.get_config_dictionary()
    assert conf["licence"] == licence
    assert conf["url"] == url
    assert conf["paper_url"] == paper_url
    assert conf["firmware_url"] == firmware_url
    # Reload the dataset
    ds = Dataset.from_config(ds.path)
    conf = ds.get_config_dictionary()
    assert conf["licence"] == licence
    assert conf["url"] == url
    assert conf["paper_url"] == paper_url
    assert conf["firmware_url"] == firmware_url


def test_scaaml_version_present(tmp_path):
    ds = Dataset(**dataset_constructor_kwargs(root_path=tmp_path))
    config = ds.get_config_dictionary()
    assert "scaaml_version" in config.keys()


def test_check_key_ap(tmp_path):
    key_ap = "k"
    # Fix numpy randomness not to cause flaky tests.
    np.random.seed(42)
    dataset = Dataset(**dataset_constructor_kwargs(root_path=tmp_path,
                                                   examples_per_shard=1,
                                                   attack_points_info={
                                                       key_ap: {
                                                           "len": 16,
                                                           "max_val": 256
                                                       },
                                                   }))
    trace_len = 1024
    # Fill the dataset.
    # Two shards with the same key
    key1 = np.random.randint(256, size=16)
    dataset.new_shard(key=key1,
                      part=0,
                      split=Dataset.TRAIN_SPLIT,
                      group=0,
                      chip_id=1)
    dataset.write_example({key_ap: key1},
                          {"trace1": np.random.random(trace_len)})
    dataset.close_shard()
    key2 = np.random.randint(256, size=16)
    dataset.new_shard(key=key2,
                      part=0,
                      split=Dataset.TEST_SPLIT,
                      group=0,
                      chip_id=1)
    dataset.write_example({key_ap: key2},
                          {"trace1": np.random.random(trace_len)})
    dataset.close_shard()

    dataset.check(key_ap=key_ap)

    # Make a duplicate key
    dataset.new_shard(key=key1,
                      part=0,
                      split=Dataset.TEST_SPLIT,
                      group=0,
                      chip_id=1)
    dataset.write_example({key_ap: key1},
                          {"trace1": np.random.random(trace_len)})
    dataset.close_shard()

    with pytest.raises(ValueError) as value_error:
        dataset.check(key_ap=key_ap)
    assert "Duplicate key" in str(value_error.value)


def test_merge_with(tmp_path):
    # Fix numpy randomness not to cause flaky tests.
    np.random.seed(42)
    resulting = Dataset(**dataset_constructor_kwargs(
        root_path=tmp_path,
        examples_per_shard=1,
        shortname="resulting",
    ))
    other_ds = Dataset(**dataset_constructor_kwargs(
        root_path=tmp_path,
        examples_per_shard=1,
        shortname="other_ds",
    ))
    trace_len = 1024
    # Fill the dataset.
    # Two shards with the same key
    key1 = np.random.randint(256, size=16)
    trace = np.zeros(trace_len)
    other_ds.new_shard(key=key1,
                       part=0,
                       split=Dataset.TRAIN_SPLIT,
                       group=0,
                       chip_id=1)
    other_ds.write_example({"key": key1}, {"trace1": trace})
    trace[0] = 0.9
    other_ds.new_shard(key=key1,
                       part=1,
                       split=Dataset.TRAIN_SPLIT,
                       group=0,
                       chip_id=1)
    other_ds.write_example({"key": key1}, {"trace1": trace})
    # Two shards with a different key
    key2 = np.random.randint(256, size=16)
    other_ds.new_shard(key=key2,
                       part=0,
                       split=Dataset.TRAIN_SPLIT,
                       group=1,
                       chip_id=1)
    other_ds.write_example({"key": key2}, {"trace1": trace})
    other_ds.new_shard(key=key2,
                       part=1,
                       split=Dataset.TRAIN_SPLIT,
                       group=1,
                       chip_id=1)
    other_ds.write_example({"key": key2}, {"trace1": trace})
    other_ds.close_shard()

    resulting.merge_with(other_ds)
    assert resulting.min_values == other_ds.min_values
    assert resulting.max_values == other_ds.max_values

    yet_another_ds = Dataset(**dataset_constructor_kwargs(
        root_path=tmp_path,
        examples_per_shard=1,
        shortname="another",
    ))
    key3 = np.random.randint(256, size=16)
    trace = np.zeros(trace_len)
    trace[1] = -0.5
    yet_another_ds.new_shard(key=key3,
                             part=0,
                             split=Dataset.TEST_SPLIT,
                             group=1,
                             chip_id=1)
    yet_another_ds.write_example({"key": key3}, {"trace1": trace})
    yet_another_ds.new_shard(key=key3,
                             part=1,
                             split=Dataset.TEST_SPLIT,
                             group=1,
                             chip_id=1)
    yet_another_ds.write_example({"key": key3}, {"trace1": trace})
    yet_another_ds.close_shard()
    key4 = np.random.randint(256, size=16)
    trace = np.zeros(trace_len)
    yet_another_ds.new_shard(key=key4,
                             part=1,
                             split=Dataset.TRAIN_SPLIT,
                             group=1,
                             chip_id=1)
    yet_another_ds.write_example({"key": key4}, {"trace1": trace})
    yet_another_ds.close_shard()

    resulting.merge_with(yet_another_ds)
    assert resulting.min_values == {"trace1": -0.5}
    assert resulting.max_values == {"trace1": 0.9}
    assert resulting.examples_per_split == {
        Dataset.TRAIN_SPLIT: 5,
        Dataset.TEST_SPLIT: 2
    }
    # The right number of files.
    assert len(list((resulting.path / Dataset.TRAIN_SPLIT).glob("*"))) == 5
    assert len(list((resulting.path / Dataset.TEST_SPLIT).glob("*"))) == 2


def test_move_shards(tmp_path):
    # Fix numpy randomness not to cause flaky tests.
    np.random.seed(42)
    ds = Dataset.get_dataset(**dataset_constructor_kwargs(
        root_path=tmp_path,
        examples_per_shard=1,
    ))
    # Fill the dataset.
    # Two shards with the same key
    key1 = np.random.randint(256, size=16)
    ds.new_shard(key=key1,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=1)
    ds.write_example({"key": key1}, {"trace1": np.random.rand(1024)})
    ds.new_shard(key=key1,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=1)
    ds.write_example({"key": key1}, {"trace1": np.random.rand(1024)})
    # Two shards with a different key
    key2 = np.random.randint(256, size=16)
    ds.new_shard(key=key2,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=1,
                 chip_id=1)
    ds.write_example({"key": key2}, {"trace1": np.random.rand(1024)})
    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=1,
                 chip_id=1)
    ds.write_example({"key": key2}, {"trace1": np.random.rand(1024)})
    ds.close_shard()

    # This is ok.
    ds.move_shards(from_split=Dataset.TRAIN_SPLIT,
                   to_split=Dataset.TEST_SPLIT,
                   shards={2, 3})
    assert len(ds.shards_list[Dataset.TRAIN_SPLIT]) == 2
    assert len(ds.shards_list[Dataset.TEST_SPLIT]) == 2
    # Move all shards back.
    ds.move_shards(from_split=Dataset.TEST_SPLIT,
                   to_split=Dataset.TRAIN_SPLIT,
                   shards=2)
    assert len(ds.shards_list[Dataset.TRAIN_SPLIT]) == 4
    assert len(ds.shards_list[Dataset.TEST_SPLIT]) == 0
    # Move nothing.
    ds.move_shards(from_split=Dataset.TEST_SPLIT,
                   to_split=Dataset.TRAIN_SPLIT,
                   shards=0)
    assert len(ds.shards_list[Dataset.TRAIN_SPLIT]) == 4
    assert len(ds.shards_list[Dataset.TEST_SPLIT]) == 0
    # Move just one, so that check fails on having the same key in test and
    # train.
    with pytest.raises(ValueError) as value_error:
        ds.move_shards(from_split=Dataset.TRAIN_SPLIT,
                       to_split=Dataset.TEST_SPLIT,
                       shards=1)
    assert "Duplicate key" in str(value_error.value)


def same_examples(ds1, ds2):
    # Check that there are the same number of examples
    config1 = ds1.get_config_dictionary()
    config2 = ds1.get_config_dictionary()
    assert config1["examples_per_split"] == config2["examples_per_split"]

    def example_iterator(dataset, split):
        """Return iterator of examples contained in a dataset."""
        config = dataset.get_config_dictionary()
        from itertools import chain
        # Concatenate all examples that are returned by Dataset.inspect.
        return chain.from_iterable(
            Dataset.inspect(dataset_path=dataset.path,
                            split=split,
                            shard_id=i,
                            num_example=dataset.examples_per_shard,
                            verbose=False).as_numpy_iterator()
            for i in range(len(config["shards_list"][split])))

    for split in config1["shards_list"].keys():
        ei1 = example_iterator(dataset=ds1, split=split)
        ei2 = example_iterator(dataset=ds2, split=split)
        # Assert that all examples (represented as numpy) are the same.
        for e1, e2 in zip(ei1, ei2):
            # Each example is a dictionary containing a "key" and "trace1"
            assert e1.keys() == e2.keys()
            for k, v in e1.items():
                assert np.isclose(v, e2[k]).all()


def test_reshape_into_new_dataset_filled(tmp_path):
    # Fix numpy randomness not to cause flaky tests.
    # Test this function here so that we do not have to redefine
    # dataset_constructor_kwargs.
    np.random.seed(42)

    old_examples_per_shard = 16
    old_ds = Dataset(**dataset_constructor_kwargs(
        root_path=tmp_path,
        examples_per_shard=old_examples_per_shard,
    ))
    # Fill in the old dataset.
    key = np.random.randint(256, size=16)
    old_ds.new_shard(key=key,
                     part=0,
                     split=Dataset.TRAIN_SPLIT,
                     group=0,
                     chip_id=1)
    for _ in range(old_examples_per_shard):
        old_ds.write_example({"key": np.random.randint(256, size=16)},
                             {"trace1": np.random.rand(1024)})
    old_ds.close_shard()

    new_ds = reshape_into_new_dataset(old_ds=old_ds, examples_per_shard=4)
    old_ds.check()
    new_ds.check()
    same_examples(old_ds, new_ds)


def test_shard_metadata_negative_chip_id(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": -13,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "Wrong chip_id, got" in str(value_error.value)


def test_shard_metadata_float_chip_id(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": 13.,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "Wrong chip_id, got" in str(value_error.value)


def test_shard_metadata_str_chip_id(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": "13",
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "Wrong chip_id, got" in str(value_error.value)


def test_shard_metadata_wrong_size(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size + 1,
        "chip_id": 13,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "Wrong size, got" in str(value_error.value)


def test_shard_metadata_wrong_path_k(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key="key", shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": 13,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "key does not match filename" in str(value_error.value)


def test_shard_metadata_wrong_path(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group + 1, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": 13,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "group does not match filename" in str(value_error.value)


def test_shard_metadata_extra_key(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "extra_key": "should not be here",
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": 13,
    }

    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)
    assert "Shard info keys are" in str(value_error.value)


def test_shard_metadata_missing_keys(tmp_path):
    with pytest.raises(ValueError) as value_error:
        Dataset._check_shard_metadata(shard_info={}, dataset_path=tmp_path)
    assert "Shard info keys are" in str(value_error.value)


def test_shard_metadata_ok(tmp_path):
    group = 1
    key = "BDC9C50A1B51732C56838405443BC76F"
    part = 2
    shard_file = tmp_path / Dataset._shard_name(
        shard_group=group, shard_key=key, shard_part=part)
    shard_file.write_text("content")
    si = {
        "examples": 64,
        "sha256": "abc123",
        "path": str(shard_file),
        "group": group,
        "key": key,
        "part": part,
        "size": os.stat(shard_file).st_size,
        "chip_id": 13,
    }

    Dataset._check_shard_metadata(shard_info=si, dataset_path=tmp_path)


def test_min_max_values_ok(tmp_path):
    """Test that min_values and max_values are not affected by mutable default
    parameter.
    """

    def min_max_t(min_value: float, max_value: float, root_path: Path) -> None:
        assert min_value <= max_value
        ds = Dataset(**dataset_constructor_kwargs(root_path=root_path))
        mid_point = min_value + (max_value - min_value) / 2
        trace = np.full(1024, mid_point, dtype=np.float64)
        trace[0] = min_value
        trace[1] = max_value
        key = np.zeros(16, dtype=np.uint8)
        ds.new_shard(key=key,
                     part=0,
                     split=Dataset.TRAIN_SPLIT,
                     group=0,
                     chip_id=1)
        ds.write_example({"key": key}, {"trace1": trace})
        ds.close_shard()
        config_dict = ds.get_config_dictionary()
        assert config_dict["min_values"]["trace1"] == min_value
        assert config_dict["max_values"]["trace1"] == max_value

    min_max_t(0.1, 0.7, root_path=tmp_path / "b")
    min_max_t(0.2, 0.5, root_path=tmp_path / "a")


def test_resume_capture(tmp_path):
    kwargs = dataset_constructor_kwargs(root_path=tmp_path)
    ds = Dataset.get_dataset(**kwargs)
    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.zeros(1024, dtype=np.float64)
    trace1[2] = 0.8  # max_val is written before deleting ds
    trace2 = np.zeros(1024, dtype=np.float64)
    chip_id = 1
    ds.new_shard(key=key,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()
    del ds
    ds = Dataset.get_dataset(**kwargs)
    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace2})
    ds.close_shard()
    config_dict = ds.get_config_dictionary()

    assert len(config_dict["shards_list"][Dataset.TRAIN_SPLIT]) == 2
    assert config_dict["max_values"]["trace1"] == 0.8


def test_info_file_raises(tmp_path):
    # Make the dataset directory with info.json in it (the "sn_al_ar_vimp_1" is
    # the slug).
    dpath = tmp_path / "sn_al_ar_vimp_1"
    dpath.mkdir()
    Dataset._get_config_path(dpath).write_text("exists")

    with pytest.raises(DatasetExistsError) as value_error:
        ds = Dataset(root_path=tmp_path,
                     shortname="sn",
                     architecture="ar",
                     implementation="imp",
                     algorithm="al",
                     version=1,
                     description="description",
                     url="",
                     firmware_url="some url",
                     firmware_sha256="abc123",
                     examples_per_shard=1,
                     measurements_info={},
                     attack_points_info={})
    assert ("Dataset info file exists and would be overwritten. "
            "Use instead:") in str(value_error.value)


def test_from_loaded_json(tmp_path):
    ds = Dataset(root_path=tmp_path,
                 shortname="shortname",
                 architecture="architecture",
                 implementation="implementation",
                 algorithm="algorithm",
                 version=1,
                 description="description",
                 url="",
                 firmware_url="some url",
                 firmware_sha256="abc123",
                 examples_per_shard=1,
                 measurements_info={"trace1": {
                     "type": "power",
                     "len": 1024,
                 }},
                 attack_points_info={
                     "key": {
                         "len": 16,
                         "max_val": 256
                     },
                 })
    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.random.rand(1024)
    chip_id = 1
    ds.new_shard(key=key,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()
    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.close_shard()
    config_dict = ds.get_config_dictionary()
    json_dict = json.loads(json.dumps(config_dict))

    loaded_dict = Dataset._from_loaded_json(json_dict)

    assert loaded_dict == config_dict


def test_from_config(tmp_path):
    # Create the file structure.
    d1 = Dataset(root_path=tmp_path,
                 shortname="shortname",
                 architecture="architecture",
                 implementation="implementation",
                 algorithm="algorithm",
                 version=1,
                 description="description",
                 url="",
                 firmware_url="some url",
                 firmware_sha256="abc123",
                 examples_per_shard=64,
                 measurements_info={},
                 attack_points_info={})
    old_files = set(tmp_path.glob("**/*"))

    d2 = Dataset.from_config(d1.path)

    # No new files have been created.
    new_files = set(tmp_path.glob("**/*"))
    assert old_files == new_files


def test_close_shard(tmp_path):
    measurements_info = {
        "trace1": {
            "type": "power",
            "len": 1024,
        }
    }
    attack_point_info = {
        "key": {
            "len": 16,
            "max_val": 256
        },
    }
    ds = Dataset(root_path=tmp_path,
                 shortname="short_name",
                 architecture="arch",
                 implementation="implementation",
                 algorithm="algorithm",
                 version=1,
                 description="description",
                 url="url",
                 firmware_url="url",
                 firmware_sha256="abc1234",
                 examples_per_shard=2,
                 measurements_info=measurements_info,
                 attack_points_info=attack_point_info)
    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.random.rand(1024)
    chip_id = 1

    assert ds.examples_per_group == {}
    assert ds.examples_per_split == {}
    ds.new_shard(key=key,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()
    assert ds.examples_per_group == {Dataset.TRAIN_SPLIT: {0: 2}}
    assert ds.examples_per_split == {Dataset.TRAIN_SPLIT: 2}
    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.close_shard()
    assert ds.examples_per_group == {Dataset.TRAIN_SPLIT: {0: 4}}
    assert ds.examples_per_split == {Dataset.TRAIN_SPLIT: 4}
    ds.check()


def test_from_config(tmp_path):
    # Create the file structure.
    d1 = Dataset(root_path=tmp_path,
                 shortname="shortname",
                 architecture="architecture",
                 implementation="implementation",
                 algorithm="algorithm",
                 version=1,
                 description="description",
                 url="",
                 firmware_url="url",
                 firmware_sha256="abc123",
                 examples_per_shard=64,
                 measurements_info={},
                 attack_points_info={})
    old_files = set(tmp_path.glob("**/*"))

    d2 = Dataset.from_config(d1.path)

    # No new files have been created.
    new_files = set(tmp_path.glob("**/*"))
    assert old_files == new_files


@patch.object(Path, "read_text")
@patch.object(Shard, "__init__")
@patch.object(Shard, "read")
def test_inspect(mock_shard_read, mock_shard_init, mock_read_text):
    split = Dataset.TEST_SPLIT
    shard_id = 0
    num_example = 5
    mock_shard_init.return_value = None
    config = {
        "keys_per_group": {},
        "examples_per_group": {},
        "shards_list": {
            Dataset.TEST_SPLIT: [{
                "path": "test/0_abcd_1.tfrec"
            }, {
                "path": "test/2_ef12_3.tfrec"
            }],
            Dataset.TRAIN_SPLIT: [{
                "path": "train/2_dead_0.tfrec"
            }, {
                "path": "train/3_beef_1.tfrec"
            }],
        },
        "attack_points_info": {
            "ap_info": "something"
        },
        "measurements_info": {
            "m_info": "else"
        },
        "compression": "GZIP",
    }
    mock_read_text.return_value = json.dumps(config)
    dir_dataset_ok = Path("/home/not_user")

    x = Dataset.inspect(dir_dataset_ok,
                        split=split,
                        shard_id=shard_id,
                        num_example=num_example)

    shard_filename = str(dir_dataset_ok /
                         config["shards_list"][split][shard_id]["path"])
    mock_shard_init.assert_called_once_with(
        shard_filename,
        attack_points_info=config["attack_points_info"],
        measurements_info=config["measurements_info"],
        measurement_dtype=tf.float32,
        compression=config["compression"])
    mock_shard_read.assert_called_once_with(num=num_example)
    assert x == mock_shard_read.return_value


@patch.object(siutils, "sha256sum")
def test_check_sha256sums(mock_sha256sum):
    sha_dictionary = {
        "test/0_01f6e272b933ec2b80ab53af245f7fa6_0.tfrec":
            "b1e3dc7c217b154ded5631d95d6265c6b1ad348ac4968acb1a74b9fb49c09c42",
        "test/1_f2e8de7fdbc602f96261ba5f8d182d73_0.tfrec":
            "7a9b214d76f68b4e1a9abf833314ae5909e96b6c4c9f81c7a020a63913dfc51c",
        "test/0_69a283f6b1eea6327afdb30f76e6fe30_0.tfrec":
            "f61009a4c6f5a77aa2c6da6d1882a50c3bd6345010966144d16e634ceeaeb730",
    }
    dpath = Path("/home/nobody/not_a_directory")
    mock_sha256sum.side_effect = lambda x: sha_dictionary[
        f"{x.parent.name}/{x.name}"]
    shards_list = {
        Dataset.TEST_SPLIT: [{
            "path": f,
            "sha256": s,
        } for f, s in sha_dictionary.items()],
    }
    pbar = lambda *args, **kwargs: args[0]
    Dataset._check_sha256sums(shards_list=shards_list, dpath=dpath, pbar=pbar)

    mock_sha256sum.side_effect = lambda _: "abcd"
    with pytest.raises(ValueError) as sha_error:
        Dataset._check_sha256sums(shards_list=shards_list,
                                  dpath=dpath,
                                  pbar=pbar)
    assert "SHA256 miss-match" in str(sha_error.value)


def test_check_metadata():
    config = {
        "shards_list": {
            Dataset.TEST_SPLIT: [{
                "examples": 64,
                "group": 0,
                "key": "01F6E272B933EC2B80AB53AF245F7FA6",
                "part": 0,
            }, {
                "examples": 64,
                "group": 1,
                "key": "F2E8DE7FDBC602F96261BA5F8D182D73",
                "part": 0,
            }, {
                "examples": 64,
                "group": 0,
                "key": "69A283F6B1EEA6327AFDB30F76E6FE30",
                "part": 0,
            }],
            Dataset.TRAIN_SPLIT: [
                {
                    "examples": 64,
                    "group": 0,
                    "key": "A4F6C39380E6D85CD2D4D5BD7EED11A8",
                    "part": 0,
                },
                {
                    "examples": 64,
                    "group": 0,
                    "key": "A4F6C39380E6D85CD2D4D5BD7EED11A8",
                    "part": 1,
                },
            ]
        },
        "examples_per_group": {
            Dataset.TEST_SPLIT: {
                "0": 2 * 64,
                "1": 1 * 64,
            },
            Dataset.TRAIN_SPLIT: {
                "0": 2 * 64
            }
        },
        "examples_per_split": {
            Dataset.TEST_SPLIT: 3 * 64,
            Dataset.TRAIN_SPLIT: 2 * 64
        },
        "examples_per_shard": 64,
    }
    Dataset._check_metadata(config=config)

    # Wrong number of examples_per_split
    bad_config = copy.deepcopy(config)
    bad_config["examples_per_split"][Dataset.TEST_SPLIT] = 5
    with pytest.raises(ValueError) as metadata_error:
        Dataset._check_metadata(config=bad_config)
    assert "Num shards in shard_list !=" in str(metadata_error.value)

    # Wrong number of examples_per_group
    bad_config = copy.deepcopy(config)
    bad_config["examples_per_group"][Dataset.TEST_SPLIT]["0"] = 5
    with pytest.raises(ValueError) as metadata_error:
        Dataset._check_metadata(config=bad_config)
    assert "Wrong sum of examples_per_group in" in str(metadata_error.value)

    # Not constant number of examples in shards
    bad_config = copy.deepcopy(config)
    bad_config["shards_list"][Dataset.TEST_SPLIT][0]["examples"] = 63
    bad_config["shards_list"][Dataset.TEST_SPLIT][2]["examples"] = 65
    with pytest.raises(ValueError) as metadata_error:
        Dataset._check_metadata(config=bad_config,
                                n_examples_in_each_shard_is_constant=True)
    assert "contain the same number of examples" in str(metadata_error.value)

    # Wrong number of examples_per_group
    bad_config = copy.deepcopy(config)
    bad_config["examples_per_group"][Dataset.TEST_SPLIT]["0"] = 127
    bad_config["examples_per_group"][Dataset.TEST_SPLIT]["1"] = 65
    with pytest.raises(ValueError) as metadata_error:
        Dataset._check_metadata(config=bad_config)
    assert "Wrong examples_per_group in" in str(metadata_error.value)


def test_shallow_check():
    pbar = lambda *args, **kwargs: args[0]
    seen_keys = set()
    train_shards = [
        {
            "key": "FFE8"
        },
    ]
    Dataset._shallow_check(seen_keys, train_shards, pbar)

    seen_keys.add(np.array([255, 232], dtype=np.uint8).tobytes())
    with pytest.raises(ValueError) as intersection_error:
        Dataset._shallow_check(seen_keys, train_shards, pbar)
    assert "Duplicate key" in str(intersection_error.value)


@patch.object(Dataset, "inspect")
def test_deep_check(mock_inspect):
    mock_inspect.return_value.as_numpy_iterator.return_value = (
        {
            "key": np.array([0, 1, 2, 255])
        },
        {
            "key": np.array([3, 1, 4, 1])
        },
    )
    seen_keys = set()
    pbar = lambda *args, **kwargs: args[0]
    train_shards = [
        {},
    ]
    Dataset._deep_check(seen_keys=seen_keys,
                        dpath="/home/not_user/not_a_directory",
                        train_shards=train_shards,
                        examples_per_shard=64,
                        pbar=pbar,
                        key_ap="key")

    seen_keys.add(np.array([3, 1, 4, 1], dtype=np.uint8).tobytes())
    with pytest.raises(ValueError) as intersection_error:
        Dataset._deep_check(seen_keys=seen_keys,
                            dpath="/home/not_user/not_a_directory",
                            train_shards=train_shards,
                            examples_per_shard=64,
                            pbar=pbar,
                            key_ap="key")
    assert "Duplicate key" in str(intersection_error.value)


def test_basic_workflow(tmp_path):
    root_path = tmp_path
    architecture = "arch"
    implementation = "implementation"
    algorithm = "algo"
    version = 1
    measurements_info = {
        # test missing measurement raise value
        # test extra measurement raise value
        "trace1": {
            "type": "power",
            "len": 1024,
        }
    }
    attack_point_info = {
        "key": {
            "len": 16,
            "max_val": 256
        },

        # test missing attack point raise value
        # test extra attack point raise value
        # "sub_byte_in": {
        #     "len": 16,
        #     "max_val": 256
        # }
    }

    shortname = "SHORTNAME"
    description = "this is a test"
    url = "https://"
    example_per_shard = 1
    fw_sha256 = "A2424512D"
    key = np.random.randint(0, 255, 16)
    key2 = np.random.randint(0, 255, 16)
    trace1 = np.random.rand(1024)

    ds = Dataset(root_path=root_path,
                 shortname=shortname,
                 architecture=architecture,
                 implementation=implementation,
                 algorithm=algorithm,
                 version=version,
                 description=description,
                 url=url,
                 firmware_url="some url",
                 firmware_sha256=fw_sha256,
                 examples_per_shard=example_per_shard,
                 measurements_info=measurements_info,
                 attack_points_info=attack_point_info)

    chip_id = 1
    ds.new_shard(key=key,
                 part=0,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key}, {"trace1": trace1})
    ds.close_shard()

    # 256 keys - with uniform bytes

    ds.new_shard(key=key2,
                 part=1,
                 split=Dataset.TRAIN_SPLIT,
                 group=0,
                 chip_id=chip_id)
    ds.write_example({"key": key2}, {"trace1": trace1})
    ds.close_shard()

    # check dataset integrity and consistency
    ds.check()
    slug = ds.slug
    # reload
    ds2 = Dataset.from_config(root_path / slug)
    ds2.inspect(root_path / slug, Dataset.TRAIN_SPLIT, 0, 1)
    ds2.summary(root_path / slug)


def test_cleanup_shards(tmp_path):

    def shard_info(group: int, key: str, part: int):
        return {
            "path": Dataset._shard_name(shard_group=group,
                                        shard_key=key,
                                        shard_part=part),
            "examples": 64,
            "size": 811345,
            "sha256": "beef",
            "group": group,
            "key": key,
            "part": part,
            "chip_id": 13,
        }  # yapf: disable

    old_config = {  # Some fields omitted.
        "licence": "some_licence",
        "paper_url": "some_url",
        "firmware_url": "fm_url",
        "examples_per_shard": 64,
        "examples_per_group": {
            Dataset.TEST_SPLIT: {
                0: 2 * 64,
                1: 1 * 64,
                2: 1 * 64,
                3: 2 * 64,
            },
            Dataset.TRAIN_SPLIT: {
                0: 5 * 64,
            }
        },
        "examples_per_split": {
            Dataset.TEST_SPLIT: 6 * 64,
            Dataset.TRAIN_SPLIT: 5 * 64,
        },
        "keys_per_split": {
            Dataset.TEST_SPLIT: 4,
            Dataset.TRAIN_SPLIT: 4,
        },
        "keys_per_group": {
            Dataset.TEST_SPLIT: {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
            },
            Dataset.TRAIN_SPLIT: {
                0: 4,
            }
        },
        "shards_list": {
            Dataset.TEST_SPLIT: [
                shard_info(group=0, key="KEY1", part=0),
                shard_info(group=0, key="KEY1", part=2),  # del
                shard_info(group=1, key="KEY2", part=2),  # del
                shard_info(group=2, key="KEY3", part=2),
                shard_info(group=3, key="KEY4", part=1),
                shard_info(group=3, key="KEY4", part=2),
            ],
            Dataset.TRAIN_SPLIT: [
                shard_info(group=0, key="keyA", part=2),  # del
                shard_info(group=0, key="keyA", part=1),
                shard_info(group=0, key="keyB", part=3),
                shard_info(group=0, key="keyC", part=4),  # del
                shard_info(group=0, key="keyD", part=5),
            ],
        },
    }
    # Populate the mock database
    Dataset._get_config_path(tmp_path).write_text(json.dumps(old_config))
    for s in Dataset.SPLITS:
        (tmp_path / s).mkdir()

    new_config = copy.deepcopy(old_config)
    # Delete some files. Remember to update "examples_per_group" and "example_per_shard".
    for i in sorted([0, 3], reverse=True):  # Remove in descending order.
        del new_config["shards_list"][Dataset.TRAIN_SPLIT][i]
    new_config["examples_per_group"][Dataset.TRAIN_SPLIT] = {
        0: 3 * 64,
    }
    new_config["examples_per_split"][Dataset.TRAIN_SPLIT] = 3 * 64
    new_config["keys_per_split"][Dataset.TRAIN_SPLIT] = 3
    new_config["keys_per_group"][Dataset.TRAIN_SPLIT] = {
        0: 3,
    }
    for i in sorted([1, 2], reverse=True):  # Remove in descending order.
        del new_config["shards_list"][Dataset.TEST_SPLIT][i]
    new_config["examples_per_group"][Dataset.TEST_SPLIT] = {
        0: 1 * 64,
        1: 0 * 64,
        2: 1 * 64,
        3: 2 * 64,
    }
    new_config["examples_per_split"][Dataset.TEST_SPLIT] = 4 * 64
    new_config["keys_per_split"][Dataset.TEST_SPLIT] = 3
    new_config["keys_per_group"][Dataset.TEST_SPLIT] = {
        0: 1,
        1: 0,
        2: 1,
        3: 1,
    }
    for i in []:  # Fill this split in old_config first
        del new_config["shards_list"][Dataset.HOLDOUT_SPLIT][i]
    # Create existing files.
    for s in new_config["shards_list"]:
        for f in new_config["shards_list"][s]:
            (tmp_path / f["path"]).touch()
    # Other files should be neither deleted nor added to the dataset.
    other_files = [
        tmp_path / "i_am_not_here.txt",
        tmp_path / Dataset.TRAIN_SPLIT / "not_a_shard.tfrec",
    ]
    for f in other_files:
        f.touch()

    corrected_config = Dataset._cleanup_shards(tmp_path, print_info=False)

    # New config is ok.
    # Loop for better readability.
    for k in corrected_config:
        assert corrected_config[k] == new_config[k], f"{k} is different"
    # Test all (so far tested that corrected_config is a subset of new_config).
    assert corrected_config == new_config
    # Other files still present.
    for f in other_files:
        assert f.exists()


def test_shard_info_from_name():
    assert Dataset._shard_info_from_name(
        "1_c80b174b5ce880a3557db2152598cafe_2.tfrec") == {
            "shard_group": 1,
            "shard_key": "c80b174b5ce880a3557db2152598cafe",
            "shard_part": 2,
        }


def test_shard_info_from_name_directory():
    assert Dataset._shard_info_from_name(
        "some/directory/1_c80b174b5ce880a3557db2152598cafe_2.tfrec") == {
            "shard_group": 1,
            "shard_key": "c80b174b5ce880a3557db2152598cafe",
            "shard_part": 2,
        }
    assert Dataset._shard_info_from_name(
        "win_dir\\1_c80b174b5ce880a3557db2152598cafe_2.tfrec") == {
            "shard_group": 1,
            "shard_key": "c80b174b5ce880a3557db2152598cafe",
            "shard_part": 2,
        }


def test_shard_name():
    assert Dataset._shard_name(
        shard_group=1,
        shard_key="c80b174b5ce880a3557db2152598cafe",
        shard_part=2) == "1_c80b174b5ce880a3557db2152598cafe_2.tfrec"


def test_shard_info_from_name_identity():
    tests = [
        {
            "shard_group": 1,
            "shard_key": "cafe",
            "shard_part": 0,
        },
        {
            "shard_group": 0,
            "shard_key": "dead",
            "shard_part": 1,
        },
        {
            "shard_group": 2,
            "shard_key": "beef",
            "shard_part": 4,
        },
        {
            "shard_group": 3,
            "shard_key": "0123",
            "shard_part": 2,
        },
        {
            "shard_group": 4,
            "shard_key": "c0de",
            "shard_part": 3,
        },
    ]
    for t in tests:
        assert Dataset._shard_info_from_name(Dataset._shard_name(**t)) == t


@patch.object(tf.io, "parse_single_example")
@patch.object(tf.data.Dataset, "interleave")
@patch.object(Dataset, "from_config")
def test_as_tfdataset_same_apname_different(mock_from_config, mock_interleave,
                                            mock_parse_single_example):
    """Test multiple attack points with the same name have content."""
    # deterministic test
    rng = np.random.default_rng(42)
    tf.random.set_seed(13)

    mock_from_config.return_value.measurements_info = {
        "trace1": {
            "type": "power",
            "len": 1024,
        },
    }
    mock_from_config.return_value.attack_points_info = {
        "key": {
            "len": 16,
            "max_val": 256,
        },
    }
    mock_from_config.return_value.shards_list = {
        Dataset.TRAIN_SPLIT: [{
            "path": "file1",
        }],
    }
    mock_from_config.return_value.keys_per_split = {
        Dataset.TRAIN_SPLIT: 42,
    }
    mock_from_config.return_value.measurement_dtype = tf.float32

    mock_interleave.return_value = tf.data.Dataset.range(10).map(
        lambda i: {
            "trace1":
                tf.random.uniform(
                    minval=0, maxval=1, shape=(1024,), dtype=tf.float32),
            "key":
                tf.random.uniform(
                    minval=0, maxval=256, shape=(16,), dtype=tf.int64),
        })

    mock_parse_single_example.side_effect = lambda tfrec, tffeatures: tfrec

    test_ds, inputs, outputs = Dataset.as_tfdataset(
        dataset_path="/tmp",
        split=Dataset.TRAIN_SPLIT,
        attack_points=[
            {
                "name": "key",
                "index": 1,
                "type": "byte",
            },
            {
                "name": "key",
                "index": 3,
                "type": "byte",
            },
        ],
        traces="trace1",
    )

    assert outputs == {
        "key_1": {
            "ap": "key",
            "byte": 1,  # This would be changed to 3 due to state sharing
            "len": 16,
            "max_val": 256,
            "type": "byte",
        },
        "key_3": {
            "ap": "key",
            "byte": 3,
            "len": 16,
            "max_val": 256,
            "type": "byte",
        },
    }

    for batch in test_ds.take(1):
        example_id = 0
        assert not np.allclose(batch[1]["key_1"][example_id],
                               batch[1]["key_3"][example_id])


def test_write_config(tmp_path):
    """Test public method of writing config."""
    kwargs = dataset_constructor_kwargs(root_path=tmp_path)
    ds = Dataset.get_dataset(**kwargs)

    ds.capture_info["testing_value"] = 42
    ds.write_config()

    ds = Dataset.from_config(ds.path)
    assert ds.capture_info["testing_value"] == 42
