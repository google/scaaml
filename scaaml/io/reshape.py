# Copyright 2021-2024 Google LLC
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
"""Build and load tensorFlow dataset Record wrapper"""

from tqdm import tqdm

from scaaml.io import Dataset
from scaaml.io import DatasetFiller


def reshape_into_new_dataset(*,
                             old_ds: Dataset,
                             examples_per_shard: int,
                             name_prefix: str = "reshaped",
                             from_idx: int = 0,
                             to_idx: int = 0,
                             url: str = "") -> Dataset:
    """Reshape each shard to have only examples_per_shard. This method
    should not change the original dataset (old_ds).

    Reshaping can take a lot of time and scaaml.io.Dataset is not
    thread-safe yet. One may write a Python script invoking
    reshape_into_new_dataset, call the script multiple times (with
    appropriate values of from_idx and to_idx), and then use
    Dataset.merge_with to merge the resulting reshaped datasets.

    Args:
      examples_per_shard: Number of examples in each shard in the new
        dataset. Needs to divide the old examples_per_shard.
      name_prefix: shortname is prefixed with the string
        f"{name_prefix}_{from_idx}_{to_idx}_" (the resulting dataset should
        not exist).
      from_idx: The index of the first shard that is reshaped (shards with
        indices in range(from_idx, to_idx) are reshaped). This holds for
        each split (it is safe to have to_idx larger than the length of
        shards_list for some/all split).
      to_idx: The index of the first shard that is not reshaped. Zero value
        stands for max(len(old_ds.shards_list[s]) for s in old_ds.shards_list)).
      url: Download URL of the new dataset.

    Raises:
      ValueError: If examples_per_shard does not divide old value of
        examples_per_shard.
      DatasetExistsError: If the new dataset already exists (change the
        name_prefix).

    Returns: The new dataset object.

    Bugs:
      Part id might be too high (max is old_ds.MAX_PART_NUMBER).

      When splitting a single shard, the key info gets distributed to the
      resulting sub-shards. When a shard contains multiple keys, this means
      that the key in the info might not even be present in the shard.
    """
    # Set proper value for to_idx.
    if to_idx == 0:
        to_idx = max(len(old_ds.shards_list[s]) for s in old_ds.shards_list)
    assert 0 <= from_idx <= to_idx
    # Check divisibility.
    if old_ds.examples_per_shard % examples_per_shard:
        raise ValueError(f"Cannot split shards with "
                         f"{old_ds.examples_per_shard} examples (=traces) "
                         f"into shards of {examples_per_shard} examples.")
    # Create new dataset, raise if it already exists.
    new_dataset = Dataset(
        root_path=old_ds.root_path,
        shortname=f"{name_prefix}_{from_idx}_{to_idx}_{old_ds.shortname}",
        architecture=old_ds.architecture,
        implementation=old_ds.implementation,
        algorithm=old_ds.algorithm,
        version=old_ds.version,
        firmware_sha256=old_ds.firmware_sha256,
        description=old_ds.description,
        examples_per_shard=examples_per_shard,  # New value.
        measurements_info=old_ds.measurements_info,
        attack_points_info=old_ds.attack_points_info,
        url=url,  # Download url should not be the same.
        firmware_url=old_ds.firmware_url,
        paper_url=old_ds.paper_url,
        licence=old_ds.licence,
        compression=old_ds.compression,
        capture_info=old_ds.capture_info,
        from_config=False)

    # The old config dictionary.
    config = old_ds.get_config_dictionary()

    # Reshape each shard, keeping the group id, incrementing the part id.
    for split in config["keys_per_split"]:
        # Deduce how many examples there are with the same key and use that
        # as plaintexts_per_key while setting repetitions to 1. This allows
        # to correctly and automatically determine the part id.
        max_part_id: int = max(
            int(shard["part"])  # Old datasets had string part id.
            for shard in config["shards_list"][split])
        examples_with_same_key: int = old_ds.examples_per_shard * (1 +
                                                                   max_part_id)
        # How many examples to skip.
        examples_to_skip: int = from_idx * old_ds.examples_per_shard

        # Context manager properly opens and closes shards.
        with DatasetFiller(
                dataset=new_dataset,
                plaintexts_per_key=examples_with_same_key,
                repetitions=1,
                skip_examples=examples_to_skip,
        ) as dataset_filler:
            # Add all shards.
            for shard_idx in tqdm(range(from_idx, to_idx),
                                  desc=f"Reshaping {split}."):
                if shard_idx >= len(config["shards_list"][split]):
                    break  # This split does not have so many shards.
                for example in Dataset.inspect(
                        dataset_path=old_ds.path,
                        split=split,
                        shard_id=shard_idx,
                        num_example=old_ds.examples_per_shard,
                        verbose=False,
                ).as_numpy_iterator():
                    # Get current key.
                    shard = config["shards_list"][split][shard_idx]
                    k = shard["key"].lower()
                    cur_key = [
                        int(k[2 * i:2 * i + 2], 16) for i in range(len(k) // 2)
                    ]

                    # Get attack points and measurement.
                    attack_points = {
                        # Call tolist first in order to avoid silently
                        # representing large numbers as multiple bytes. Large
                        # values fail loud.
                        # TODO allow saving other data types than bytearray.
                        ap_name: bytearray(example[ap_name].tolist())
                        for ap_name in old_ds.attack_points_info
                    }
                    measurement = {
                        m_name: example[m_name]
                        for m_name in old_ds.measurements_info
                    }
                    # Check that the lengths and max values are as expected.
                    for ap_name, ap_val in attack_points.items():
                        ap_info = old_ds.attack_points_info
                        assert len(ap_val) == ap_info[ap_name]["len"]
                        assert max(ap_val) < ap_info[ap_name]["max_val"]

                    # Write the example (open new shards automatically).
                    dataset_filler.write_example(
                        attack_points=attack_points,
                        measurement=measurement,
                        current_key=cur_key,
                        split_name=split,
                        chip_id=shard["chip_id"],
                    )
    # Check the new dataset.
    new_dataset.check()
    return new_dataset
