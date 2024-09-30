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

import copy
import math
import json
import os
from collections import defaultdict
import shutil
from time import time
from typing_extensions import TypeAlias
from typing import Any, DefaultDict, Dict, Generic, Iterable, Iterator
from typing import List, Literal, Optional
from typing import Union, Sequence, Set, Tuple, Type, TypeVar, cast
from pathlib import Path
import pprint

from tabulate import tabulate
from termcolor import cprint
from tqdm.auto import tqdm
import numpy as np
import semver
import tensorflow as tf

import scaaml
from scaaml.utils import bytelist_to_hex
from scaaml.io.spell_check import find_misspellings
import scaaml.io.utils as siutils
from scaaml.io.utils import dtype_name_to_dtype, dtype_dtype_to_name
from scaaml.io.shard import Shard, CompressionT
from scaaml.io.errors import DatasetExistsError

_T = TypeVar("_T")


class FakeTqdm(Generic[_T]):

    def __init__(self, iterable: Iterable[_T], **kwargs: Any) -> None:
        del kwargs
        self.iterable = iterable

    def __iter__(self) -> Iterator[_T]:
        return iter(self.iterable)


ProgressBarT: TypeAlias = Union[
    Type[tqdm],  # type: ignore[type-arg]
    Type[FakeTqdm[_T]]]


class Dataset:
    """Dataset class."""
    # Valid split values (used also as directory names).
    SPLIT_T: TypeAlias = Literal["train", "test", "holdout"]
    TRAIN_SPLIT: SPLIT_T = "train"
    TEST_SPLIT: SPLIT_T = "test"
    HOLDOUT_SPLIT: SPLIT_T = "holdout"
    SPLITS: Tuple[SPLIT_T, SPLIT_T,
                  SPLIT_T] = (TRAIN_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT)
    # Largest possible part number.
    MAX_PART_NUMBER = 10

    def __init__(
        self,
        *,
        root_path: str,
        shortname: str,
        architecture: str,
        implementation: str,
        algorithm: str,
        version: int,
        firmware_sha256: str,
        description: str,
        examples_per_shard: int,
        measurements_info: Dict[str, Any],
        attack_points_info: Dict[str, Dict[str, Any]],
        url: str,
        firmware_url: str = "",
        paper_url: str = "",
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        compression: CompressionT = "GZIP",
        shards_list: Optional[Dict[str, List[Any]]] = None,
        keys_per_group: Optional[Dict[str, Dict[int, int]]] = None,
        keys_per_split: Optional[Dict[str, int]] = None,
        examples_per_group: Optional[Dict[str, Dict[int, int]]] = None,
        examples_per_split: Optional[Dict[str, int]] = None,
        capture_info: Optional[Dict[str, Any]] = None,
        min_values: Optional[Dict[str, float]] = None,
        max_values: Optional[Dict[str, float]] = None,
        from_config: bool = False,
        verbose: bool = True,
        measurement_dtype: tf.dtypes.DType = tf.float32,
    ) -> None:
        """Class for saving and loading a database.

        Args:
          url: Where to download this dataset.
          firmware_url: Where to download the firmware used while capture.
          paper_url: Where to find the published paper.
          licence: URL or the whole licence the dataset is published under.
          from_config: This Dataset object has been created from a saved
            config, root_path thus points to what should be self.path. When
            True set self.path = root_path, self.root_path to be the parent of
            self.path. In this case it does not necessarily hold that
            self.path.name == self.slug (the directory could have been renamed).
          measurement_dtype (tf.dtypes.DType): Data type of traces in shards.
            Can be either tf.float32 ro tf.float16.
          verbose: If True, print the dataset path.

        Raises:
          ValueError: If firmware_sha256 or firmware_url evaluates to False.
          DatasetExistsError: If creating this object would overwrite the
            corresponding config file.
        """
        # Do not allow mutable default parameters.
        if capture_info is None:
            capture_info = {}
        if min_values is None:
            min_values = {}
        if max_values is None:
            max_values = {}
        self.shortname = shortname
        self.architecture = architecture
        self.implementation = implementation
        self.algorithm = algorithm
        self.version = version
        self.compression = compression
        self.firmware_sha256 = firmware_sha256
        self.description = description
        self.url = url
        self.firmware_url = firmware_url
        self.paper_url = paper_url
        self.licence = licence

        self.capture_info = capture_info or {}
        self.measurements_info = measurements_info
        self.attack_points_info = attack_points_info
        self.measurement_dtype = measurement_dtype

        if not self.firmware_sha256:
            raise ValueError("Firmware hash is required")
        if not self.firmware_url:
            raise ValueError("Firmware URL is required")

        self.slug = (f"{shortname}_{algorithm}_{architecture}_"
                     f"v{implementation}_{version}")
        if from_config:
            self.path = Path(root_path)
            self.root_path = str(self.path.parent)
        else:
            self.root_path = root_path
            self.path = Path(self.root_path) / self.slug
            # create directory -- check if its empty
            if Dataset._get_config_path(self.path).exists():
                raise DatasetExistsError(dataset_path=self.path)
            else:
                # create path if needed
                self.path.mkdir(parents=True)
                Path(self.path / Dataset.TRAIN_SPLIT).mkdir()
                Path(self.path / Dataset.TEST_SPLIT).mkdir()
                Path(self.path / Dataset.HOLDOUT_SPLIT).mkdir()

        if verbose:
            cprint(f"Dataset path: {self.path}", "green")

        # current shard tracking
        self.shard_key: Optional[str] = None
        # track key change for counting
        self.prev_shard_key: Optional[str] = None
        self.shard_path: Optional[str] = None
        self.shard_split: Optional[str] = None
        self.shard_part: Optional[int] = None
        self.shard_relative_path: Optional[str] = None  # for the shard list
        self.curr_shard: Optional[Shard] = None  # current_ shard object

        # [counters] - must be passed as param to allow reload.
        # shards_list[split] is a list of shard info dictionaries (where split
        # in Dataset.SPLITS).
        self.shards_list: DefaultDict[str, List[Any]]
        self.shards_list = siutils.ddict(
            value=shards_list,  # type: ignore[arg-type]
            levels=1,
            type_var=list)

        # keys counting
        # keys_per_group[split][group_id] contains the number (int) of keys
        # belonging to the group (group_id is int)
        self.keys_per_group = siutils.ddict(
            value=keys_per_group,  # type: ignore[arg-type]
            levels=2,
            type_var=int)
        self.keys_per_split = siutils.ddict(
            value=keys_per_split,  # type: ignore[arg-type]
            levels=1,
            type_var=int)

        # examples counting
        # keys_per_group[split][gid] = cnt
        self.examples_per_group = siutils.ddict(
            value=examples_per_group,  # type: ignore[arg-type]
            levels=2,
            type_var=int)
        self.examples_per_split = siutils.ddict(
            value=examples_per_split,  # type: ignore[arg-type]
            levels=1,
            type_var=int)
        self.examples_per_shard = examples_per_shard

        # traces extreme values
        self.min_values = min_values or {}
        self.max_values = max_values or {}
        for k in measurements_info.keys():
            # init only if not existing
            if k not in self.min_values:
                self.min_values[k] = math.inf
                self.max_values[k] = 0.

        # write config if needed
        if not from_config:
            self._write_config()

    @staticmethod
    def get_dataset(*args: Any, **kwargs: Any) -> "Dataset":
        """Convenience method for getting a Dataset either by creating a new
        dataset using the Dataset constructor or by calling Dataset.from_config.

        Args: Same as scaaml.io.Dataset.__init__

        Returns: A scaaml.io.Dataset object.

        Raises: ValueError if the dataset version is higher than the scaaml
          module used (via Dataset.from_config).
        """
        try:
            return Dataset(*args, **kwargs)
        except DatasetExistsError as err:
            return Dataset.from_config(dataset_path=str(err.dataset_path))

    @staticmethod
    def _shard_name(shard_group: int, shard_key: str, shard_part: int) -> str:
        """Return filename of the shard. When updating this method also update
        Dataset._shard_info_from_name.

        Args:
          shard_group: The group this shard belongs to.
          shard_key: The key contained in this shard (hex encoded).
          shard_part: The part this shard belongs to.

        Returns: Lowercase filename of the shard (including .tfrec filetype).
        """
        sname = f"{shard_group}_{shard_key}_{shard_part}.tfrec"
        return sname.lower()

    @staticmethod
    def _shard_info_from_name(shard_name: str) -> Dict[str, Union[int, str]]:
        """Inverse of Dataset._shard_name. This method is used by
        Dataset.cleanup_shards to count how many shards per group or part are
        left.

        Args:
          shard_name: The filename of the shard as returned by
            Dataset._shard_name or with a parent directory.

        Returns: A dictionary representation of Dataset._shard_name kwargs.
        """
        for dir_separator in ["\\", "/"]:
            if dir_separator in shard_name:
                shard_name = shard_name.split(dir_separator)[-1]
        parts = shard_name.split("_")
        kwargs: Dict[str, Union[int, str]] = {}
        kwargs["shard_group"] = int(parts[0])
        kwargs["shard_key"] = parts[1]
        kwargs["shard_part"] = int(parts[2].split(".")[0])
        return kwargs

    def new_shard(self, *, key: Sequence[int], part: int, group: int,
                  split: str, chip_id: int) -> None:
        """Initiate a new key

        Args:
            key: the key that was used to create the measurements.

            part: Indicate which part of a given key set of capture this
            shard represent. Capture are splitted into parts to easily
            allow to restrict the number of traces used per key.

            group: logical group the shard belong to. For example,
            on AES a group represent a collection of shard that have distinct
            byte values. It allows to balance the diversity of keys when using
            a subset of the dataset.

            split: the split the shard belongs to {train, test, holdout}

            chip_id: indicate which chip was used for collecting the traces.

        """
        # finalize previous shard if need
        if self.curr_shard:
            self.close_shard()

        if split not in Dataset.SPLITS:
            raise ValueError(f"Invalid split, must be in {Dataset.SPLITS}")

        if part < 0 or part > self.MAX_PART_NUMBER:
            raise ValueError(f"Invalid part value -- must be in "
                             f"[0, Dataset.MAX_PART_NUMBER] "
                             f"(that is [0, {self.MAX_PART_NUMBER}]).")

        self.shard_split = split
        self.shard_part = part
        self.shard_group = group
        self.shard_key = bytelist_to_hex(key, spacer="")
        self.shard_chip_id = chip_id

        # shard name
        fname = Dataset._shard_name(self.shard_group, self.shard_key,
                                    self.shard_part)
        self.shard_relative_path = f"{split}/{fname}"
        self.shard_path = str(self.path / self.shard_relative_path)

        # new shard
        self.curr_shard = Shard(self.shard_path,
                                attack_points_info=self.attack_points_info,
                                measurements_info=self.measurements_info,
                                measurement_dtype=self.measurement_dtype,
                                compression=self.compression)

    def write_example(self, attack_points: Dict[str, bytearray],
                      measurement: Dict[str, Any]) -> None:
        assert self.curr_shard is not None
        self.curr_shard.write(attack_points, measurement)

    def close_shard(self) -> None:
        # close the shard
        assert self.curr_shard is not None
        assert self.shard_path is not None
        stats = self.curr_shard.close()
        if stats.examples != self.examples_per_shard:
            cprint(
                f"This shard contains {stats.examples}, expected "
                f"{self.examples_per_shard}", "red")

        # update min/max values
        for k, v in stats.min_values.items():
            self.min_values[k] = min(self.min_values[k], v)

        for k, v in stats.max_values.items():
            self.max_values[k] = max(self.max_values[k], v)

        # update key stats only if key changed
        if self.shard_key != self.prev_shard_key:
            self.keys_per_split[self.shard_split] += 1  # type: ignore[index]
            tmp = self.keys_per_group[self.shard_split]  # type: ignore[index]
            tmp[self.shard_group] += 1  # type: ignore[index]
            self.prev_shard_key = self.shard_key

        self.examples_per_split[
            self.shard_split] += stats.examples  # type: ignore[index]
        tmp = self.examples_per_group[self.shard_split]  # type: ignore[index]
        tmp[self.shard_group] += stats.examples  # type: ignore[index]

        # record in shard list
        self.shards_list[self.shard_split].append({  # type: ignore[index]
            "path": str(self.shard_relative_path),
            "examples": stats.examples,
            "size": os.stat(self.shard_path).st_size,
            "sha256": siutils.sha256sum(self.shard_path).lower(),
            "group": self.shard_group,
            "key": self.shard_key,
            "part": self.shard_part,
            "chip_id": self.shard_chip_id
        })

        # update config
        self._write_config()
        self.curr_shard = None

    @staticmethod
    def as_tfdataset(
        dataset_path: str,
        split: SPLIT_T,
        attack_points: List[Dict[str, Union[str, int]]],
        traces: Union[List[str], str],
        *,
        shards: Optional[int] = None,
        parts: Optional[Union[List[int], int]] = None,
        trace_start: int = 0,
        trace_len: Optional[int] = None,
        batch_size: int = 32,
        prefetch: int = 4,
        file_parallelism: Optional[int] = os.cpu_count(),
        parallelism: Optional[int] = os.cpu_count(),
        shuffle: int = 1000,
        additional_attack_points: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Tuple[tf.data.Dataset[Any], Dict[str, Any], Dict[str, Any]]:
        """"Dataset as tfdataset

        Args:
          dataset_path (str): The root path of the dataset.
          split (SPLIT_T): Split, see Dataset.SPLITS.
          attack_points (List[Dict[str, Union[str, int]]]): Attack point
            information. Contains the attack point name, index, and type. For
            example:
            ```python
            [
              { "name": "key", "index": 1, "type": "byte" },
              { "name": "sub_bytes_out", "index": 0, "type": "byte" },
              { "name": "sub_bytes_out", "index": 1, "type": "byte" },
              { "name": "sub_bytes_out", "index": 2, "type": "byte" },
            ]
          ```
          traces (Union[List[str], str]): Either a single trace name or a list
            of trace names.
          shards (Optional[int]): If specified limits the dataset to the first
            `shards` shards.
          parts (Optional[Union[List[int], int]]): Not implemented.
          trace_start (int): Skip this many first points of each trace.
          trace_len (Optional[int]): Return trace of this length (more
            formally of length min(trace_len, original length - trace_start)).
            If None then trace[trace_start:] is used.
          batch_size (int): Number of examples in a single batch.
          prefetch (int): Prefetch this many batches.
          file_parallelism (Optional[int]): IO parallelism.
          parallelism (Optional[int]): Parallelism of trace decoding and
            processing (ignored if shuffle is zero).
          shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic.
          additional_attack_points (Optional[List[Dict]]): Additional attack
            points. Each attack point has two entries:
              info which contains max_val, full_name, name, index, ap
              func which is a callable which gets rec and returns int value.
          kwargs: Convenience parameter if we pass more than expected.

        FIXME: restrict shards to specific part if they exists.

        """
        del kwargs  # unused

        if parts:
            raise NotImplementedError("Implement part filtering")

        # boxing
        if isinstance(traces, str):
            traces = [traces]

        # loading info
        dpath = Path(dataset_path)
        dataset = Dataset.from_config(dataset_path)

        if split not in dataset.keys_per_split:
            raise ValueError("Unknown split -- see Dataset.summary() for list")

        # TF_FEATURES construction: must contains all features and be global
        tf_features = {}  # what is decoded
        for name, ipt in dataset.measurements_info.items():
            if dataset.measurement_dtype == tf.float32:
                tf_features[name] = tf.io.FixedLenFeature([ipt["len"]],
                                                          tf.float32)
            elif dataset.measurement_dtype == tf.float16:
                tf_features[name] = tf.io.FixedLenFeature((), tf.string)
            else:
                raise ValueError(
                    f"Wrong measurement_dtype: {dataset.measurement_dtype}")
        for name, ap in dataset.attack_points_info.items():
            tf_features[name] = tf.io.FixedLenFeature([ap["len"]], tf.int64)

        # decoding function
        def from_tfrecord(
                tfrecord: str) -> Dict[str, Union[tf.Tensor, tf.SparseTensor]]:
            rec: Dict[str, Union[tf.Tensor, tf.SparseTensor]]
            rec = tf.io.parse_single_example(tfrecord, tf_features)
            # Decoding needed for float16
            if dataset.measurement_dtype == tf.float16:
                for name, ipt in dataset.measurements_info.items():
                    rec[name] = tf.io.parse_tensor(rec[name],
                                                   out_type=tf.float16)
                    rec[name] = tf.ensure_shape(rec[name], shape=(ipt["len"],))
            elif dataset.measurement_dtype != tf.float32:
                raise ValueError(
                    f"Wrong measurement_dtype: {dataset.measurement_dtype}")
            return rec

        # inputs construction
        inputs: Dict[str, Any] = {}  # model inputs
        for name in traces:
            ipt = copy.deepcopy(dataset.measurements_info[name])
            inputs[name] = ipt

            inputs[name]["min"] = tf.constant(dataset.min_values[name])
            inputs[name]["max"] = tf.constant(dataset.max_values[name])
            delta = tf.constant(inputs[name]["max"] - inputs[name]["min"])
            inputs[name]["delta"] = delta

        # output construction
        outputs: Dict[str, Any] = {}  # model outputs
        for attack_point in attack_points:
            ap_name: str = cast(str, attack_point["name"])
            ap_index = attack_point["index"]
            ap_type = attack_point["type"]
            full_name = f"{ap_name}_{ap_index}"

            # Add attack point info (len, max_val).
            outputs[full_name] = copy.deepcopy(
                dataset.attack_points_info[ap_name])
            # Set the attack point name (keep backwards compatibility).
            outputs[full_name]["ap"] = ap_name
            # Set the byte/bit index (keep backwards compatibility).
            outputs[full_name]["byte"] = ap_index
            # Set the type of the attack point.
            outputs[full_name]["type"] = ap_type

        # processing function
        # @tf.function
        def process_record(
                rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            "process the tf record to get it ready for learning"
            x = {}
            # normalize the traces
            for name, data in inputs.items():
                trace: Any = rec[name]

                # truncate if needed
                if trace_start:
                    trace = trace[trace_start:]

                if trace_len:
                    trace = trace[:trace_len]

                # Make sure the shape is ok.
                trace = tf.ensure_shape(trace, (trace_len,))
                # Cast to float32 before feeding to neural network.
                trace = tf.cast(trace, dtype=tf.float32)

                # assign
                x[name] = trace
                inputs[name]["shape"] = trace.shape
            # Encoding the output for each ap/byte
            y = {}
            for name, data in outputs.items():
                max_val = data["max_val"]
                if max_val == 2:
                    # Binary classification.
                    v = tf.cast(rec[data["ap"]][data["byte"]], dtype=tf.float32)
                else:
                    # Multiple classes classification.
                    v = tf.one_hot(rec[data["ap"]][data["byte"]], max_val)
                y[name] = v

            # Add additional attack points
            if additional_attack_points:
                for attack_point in additional_attack_points:
                    y[attack_point["info"]["full_name"]] = tf.one_hot(
                        attack_point["func"](rec),
                        attack_point["info"]["max_val"],
                    )

            return (x, y)

        # collect and truncate shard list of a given split
        # this is done prior to anything to allow to only download the nth
        # first shards
        shards_list = dataset.shards_list[split]
        if shards:
            shards_list = shards_list[:shards]
        shards_paths = [str(dpath / s["path"]) for s in shards_list]
        num_shards = len(shards_paths)
        # print(shards_paths)
        # dataset creation
        # with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices(shards_paths)
        ds = ds.repeat()
        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            ds = ds.shuffle(num_shards)

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the faster and more compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        #
        # If shuffle is equal to zero we produce deterministic order of
        # examples. By setting cycle_length to one (and num_parallel_calls to
        # default) we also avoid non-obvious interleaving patterns when there
        # are only a few shards. Deterministic defaults in order to avoid a
        # warning (deterministic = False does nothing when there is no thread
        # pool created by num_parallel_calls).
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(
                x, compression_type=dataset.compression),  # noqa
            cycle_length=file_parallelism if shuffle else 1,
            block_length=1,
            num_parallel_calls=file_parallelism if shuffle else None,
            deterministic=False if shuffle else None,
        )
        # decode to records
        ds = ds.map(from_tfrecord, num_parallel_calls=parallelism)
        # process them
        ds = ds.map(process_record, num_parallel_calls=parallelism)

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            ds = ds.shuffle(shuffle)

        # Additional attack points to outputs
        if additional_attack_points:
            for attack_point in additional_attack_points:
                info = cast(Dict[str, Any], attack_point["info"])
                name = cast(str, info["full_name"])
                outputs[name] = info

        # # batching with repeat
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch)

        return ds, inputs, outputs

    @staticmethod
    def summary(dataset_path: Union[str, Path]) -> None:
        """Print a summary of the dataset"""
        lst = [
            "shortname", "description", "url", "architecture", "implementation",
            "algorithm", "version", "compression"
        ]

        conf_path = Dataset._get_config_path(dataset_path)
        config = Dataset._load_config(conf_path)
        cprint("[Dataset Summary]", "cyan")
        cprint("Info", "yellow")
        print(tabulate([[k, config.get(k, "")] for k in lst]))

        cprint("\nAttack Points", "yellow")
        d = [[k, v["len"], v["max_val"]]
             for k, v in config["attack_points_info"].items()]
        print(tabulate(d, headers=["ap", "len", "max_val"]))

        cprint("\nMeasurements", "magenta")
        d = [[k, v["type"], v["len"]]
             for k, v in config["measurements_info"].items()]
        print(tabulate(d, headers=["name", "type", "len"]))

        cprint("\nContent", "green")
        d = []
        for split in config["keys_per_split"].keys():
            d.append([
                split,
                len(config["shards_list"][split]),
                config["keys_per_split"][split],
                config["examples_per_split"][split],
            ])
        print(tabulate(d, ["split", "num_shards", "num_keys", "num_examples"]))

    @staticmethod
    def inspect(dataset_path: Union[str, Path],
                split: SPLIT_T,
                shard_id: int,
                num_example: int,
                verbose: bool = True) -> tf.data.Dataset[Any]:
        """Display the content of a given shard.

        Args:
          dataset_path: Root path to the dataset.
          split: The split to inspect.
          shard_id: Index into the shards_list.
          num_example: How many examples to return. If -1 or larger than
            examples_per_shard, all examples are taken.
          verbose: Print debugging output to stdout.

        Returns: tf TakeDataset object.
        """
        conf_path = Dataset._get_config_path(dataset_path)
        config = Dataset._load_config(conf_path)

        # Find out measurement_dtype, default to float32
        measurement_dtype = dtype_name_to_dtype(
            config.get("measurement_dtype", "float32"))

        shard_path = Path(
            dataset_path) / config["shards_list"][split][shard_id]["path"]
        if verbose:
            cprint(f"Reading shard {shard_path}", "cyan")
        s = Shard(str(shard_path),
                  attack_points_info=config["attack_points_info"],
                  measurements_info=config["measurements_info"],
                  compression=config["compression"],
                  measurement_dtype=measurement_dtype)
        data = s.read(num=num_example)
        if verbose:
            print(data)
        return data

    def check(self,
              deep_check: bool = True,
              show_progressbar: bool = True,
              key_ap: str = "key") -> None:
        """Check the dataset integrity. Check integrity of metadata in config
        and also that no key from the train is in the test.

        Args:
          deep_check: When checking that keys in test and train splits are
            disjoint inspect train shards (set to True if a single train shard
            may contain multiple different keys).
          show_progressbar: Use tqdm to show a progressbar for different checks.
          key_ap: The attack point that is checked for when checking
            disjointness of splits.

        Raises: ValueError if the dataset is inconsistent.
        """
        if key_ap not in self.attack_points_info:
            raise ValueError(f"{key_ap} is not an attack point.")
        pbar: ProgressBarT[Any]
        if show_progressbar:
            pbar = tqdm
        else:
            # Redefine tqdm to the identity function returning the first unnamed
            # parameter.
            pbar = FakeTqdm

        Dataset._check_chip_id(self.shards_list)

        Dataset._check_metadata(config=self.get_config_dictionary())
        Dataset._check_sha256sums(shards_list=self.shards_list,
                                  dpath=Path(self.path),
                                  pbar=pbar)
        # Check shard metadata
        for shard_list in self.shards_list.values():
            for shard_info in shard_list:
                Dataset._check_shard_metadata(shard_info=shard_info,
                                              dataset_path=self.path)
        # Ensure that no keys in the train split are present in the test split.
        has_test: bool = Dataset.TEST_SPLIT in self.examples_per_split
        has_train: bool = Dataset.TRAIN_SPLIT in self.examples_per_split
        if has_test and has_train:
            self._check_disjoint_keys(pbar=pbar,
                                      key_ap=key_ap,
                                      deep_check=deep_check)

    @staticmethod
    def _check_chip_id(shards_list: Dict[str, List[Any]]) -> None:
        """Check that no chip_id is repeated between holdout and other splits.

        Raises: a ValueError if the same chip_id is present in holdout and
          another split.
        """
        if Dataset.HOLDOUT_SPLIT not in shards_list:
            return

        holdout_chip_ids = set(
            shard_info["chip_id"]
            for shard_info in shards_list[Dataset.HOLDOUT_SPLIT])

        for split in (Dataset.TRAIN_SPLIT, Dataset.TEST_SPLIT):
            # Check if the split is present.
            if split not in shards_list:
                continue

            # Check for duplicates between split and holdout.
            for shard_info in shards_list[split]:
                if shard_info["chip_id"] in holdout_chip_ids:
                    raise ValueError(f"Same chip_id in {split} and "
                                     f"{Dataset.HOLDOUT_SPLIT}")

    def _check_disjoint_keys(self,
                             pbar: ProgressBarT[Any],
                             key_ap: str,
                             deep_check: bool = True) -> None:
        """Check that no key in the train split is present in the test split.

        Args:
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
          key_ap: The attack point that is checked for when checking
            disjointness of splits.
          deep_check: When checking that keys in test and train splits are
            disjoint inspect train shards (set to True if a single train shard
            may contain multiple different keys).

        Raises: ValueError if some key from train is present in test.
        """
        seen_keys = set()
        for i in range(len(self.shards_list[Dataset.TEST_SPLIT])):
            for example in Dataset.inspect(dataset_path=self.path,
                                           split=Dataset.TEST_SPLIT,
                                           shard_id=i,
                                           num_example=self.examples_per_shard,
                                           verbose=False).as_numpy_iterator():
                seen_keys.add(example[key_ap].astype(np.uint8).tobytes())
        if deep_check:
            Dataset._deep_check(
                seen_keys=seen_keys,
                dpath=self.path,
                train_shards=self.shards_list[Dataset.TRAIN_SPLIT],
                pbar=pbar,
                examples_per_shard=self.examples_per_shard,
                key_ap=key_ap)
        else:
            Dataset._shallow_check(
                seen_keys=seen_keys,
                train_shards=self.shards_list[Dataset.TRAIN_SPLIT],
                pbar=pbar)

    @staticmethod
    def _check_sha256sums(shards_list: Dict[str, List[Any]], dpath: Path,
                          pbar: ProgressBarT[Any]) -> None:
        """Check the metadata of this dataset.

        Args:
          shards_list: Dictionary with information about each shard.
            Use get_config_dictionary()["shards_list"]
          dpath: Root path of the dataset.
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).

        Raises: ValueError if some hash does not match.
        """
        for split, shard_list in shards_list.items():
            for shard_info in pbar(shard_list,
                                   desc=f"Checking sha for {split}"):
                shard_path = dpath / shard_info["path"]
                sha_hash = siutils.sha256sum(shard_path)
                if sha_hash != shard_info["sha256"]:
                    raise ValueError(shard_info["path"], "SHA256 miss-match")

    @staticmethod
    def _check_shard_metadata(shard_info: Dict[str, Any],
                              dataset_path: Path) -> None:
        """Checks shard metadata.

        Args:
          shard_info: Dictionary of the shard metadata.
          dataset_path: Dataset path, so that we can check size of the shard
            file.

        Raises: ValueError if the metadata is inconsistent.
        """
        # Check that only expected keys are present:
        si_keys = {
            "examples",  # Checked by Dataset._check_metadata
            "sha256",  # Checked by Dataset._check_sha256sums
            "path",  # Checked by Dataset._check_sha256sums
            "group",  # Checked against path
            "key",  # Checked against path
            "part",  # Checked against path
            "size",  # Checked here
            "chip_id",  # Checked that it is a non-negative integer
        }
        if set(shard_info.keys()) != si_keys:
            raise ValueError(f"Shard info keys are: {shard_info.keys()} "
                             f"expected: {si_keys}, in shard: {shard_info}")
        # Check that the info corresponds to the filename:
        file_info = Dataset._shard_info_from_name(shard_info["path"])
        for key in ["group", "part"]:
            # either shard_group or shard_part
            shard_membership: str = f"shard_{key}"
            if file_info[shard_membership] != shard_info[key]:
                raise ValueError(f"{key} does not match filename, expected: "
                                 f"{file_info[shard_membership]}, got: "
                                 f"{shard_info[key]}, in shard: {shard_info}")
        # Check key (in filename it is lower case, in info it is upper case)
        if str(file_info["shard_key"]).lower() != shard_info["key"].lower():
            expected_key = file_info["shard_key"]
            actual_key = shard_info["key"]
            raise ValueError(f"key does not match filename, expected: "
                             f"{expected_key}, got: {actual_key} (not case "
                             f"sensitive), in shard: {shard_info}")
        # Check size of the file
        size = os.stat(dataset_path / shard_info["path"]).st_size
        if size != shard_info["size"]:
            expected_size = shard_info["size"]
            raise ValueError(f"Wrong size, got: {size}, expected: "
                             f"{expected_size}, in shard: {shard_info}")
        # Check chip_id is non-negative integer
        chip_id = shard_info["chip_id"]
        if not isinstance(chip_id, int) or chip_id < 0:
            raise ValueError(f"Wrong chip_id, got: {chip_id}, of type: "
                             f"{type(chip_id)}, in shard: {shard_info}")

    @staticmethod
    def _check_metadata(
            config: Dict[str, Any],
            n_examples_in_each_shard_is_constant: bool = False) -> None:
        """Check the metadata of this dataset.

        Args:
          config: A dictionary representing the metadata.
          n_examples_in_each_shard_is_constant: Check that each shard contains
            exactly examples_per_shard examples.

        Raises: ValueError if some metadata do not match.
        """
        for split, expected_examples in config["examples_per_split"].items():
            shard_list = config["shards_list"][split]
            # Checking we have the right number of shards.
            n_shards: int = expected_examples // config["examples_per_shard"]
            if len(shard_list) != n_shards:
                raise ValueError("Num shards in shard_list != "
                                 "examples_per_split // examples_per_shard")
            # Check that expected_examples is a multiple of examples_per_shard.
            if expected_examples % config["examples_per_shard"]:
                raise ValueError("expected_examples is not divisible by "
                                 "examples_per_shard")

            if expected_examples != sum(s["examples"] for s in shard_list):
                raise ValueError(f"Mismatch in expected_examples, shards "
                                 f"metadata do not agree in {split}.")
            if n_examples_in_each_shard_is_constant:
                # All shards have the same number of examples.
                if any(s["examples"] != config["examples_per_shard"]
                       for s in shard_list):
                    raise ValueError(f"Not all shards in {split} contain the "
                                     f"same number of examples.")

            # Check examples_per_group sums to the right thing.
            sum_examples_per_group = sum(
                config["examples_per_group"][split].values())
            if sum_examples_per_group != expected_examples:
                raise ValueError(f"Wrong sum of examples_per_group in {split}")
            # Check examples_per_group in individual groups.
            # Dataset.check can be called either after creating a dataset (when
            # all measurements are done) or after loading from a config. The
            # JSON file-format only allows keys to be strings. When the dataset
            # is created the group ids are integers, but when dataset is loaded
            # they are strings. We check the case where all keys are strings.
            examples_per_group: Dict[str, int] = defaultdict(int)
            for shard in shard_list:
                examples_per_group[str(shard["group"])] += shard["examples"]
            examples_per_group_config = {
                str(k): v
                for k, v in config["examples_per_group"][split].items()
            }
            if examples_per_group != examples_per_group_config:
                raise ValueError(f"Wrong examples_per_group in {split}")

            actual_examples = 0
            for shard_info in shard_list:
                actual_examples += shard_info["examples"]
                if shard_info["examples"] != config["examples_per_shard"]:
                    expected_examples_per_shard = config["examples_per_shard"]
                    actual_examples_now = shard_info["examples"]
                    raise ValueError(f"Wrong number of examples, expected: "
                                     f"{expected_examples_per_shard}, got:"
                                     f"{actual_examples_now}, in shard: "
                                     f"{shard_info}")

            if actual_examples != expected_examples:
                raise ValueError("sum example don't match top_examples")

    @staticmethod
    def _shallow_check(seen_keys: Set[bytes], train_shards: List[Any],
                       pbar: ProgressBarT[Any]) -> None:
        """Check just what is in self.shards_list info (do not parse all
        shards).

        Args:
          seen_keys: Set of all keys that are present in the test split.
          train_shards: Description of train shards
            (self.shards_list[Dataset.TRAIN_SPLIT]).
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
        """
        for shard in pbar(train_shards, desc="Checking test key uniqueness"):
            k = shard["key"].lower()
            list_k = [int(k[2 * i:2 * i + 2], 16) for i in range(len(k) // 2)]
            cur_key = np.array(list_k, dtype=np.uint8).tobytes()
            if cur_key in seen_keys:
                raise ValueError(
                    f"Duplicate key: {k} in test split, in {shard}")

    @staticmethod
    def _deep_check(*, seen_keys: Set[bytes], dpath: Path,
                    train_shards: List[Any], pbar: ProgressBarT[Any],
                    examples_per_shard: int, key_ap: str) -> None:
        """Check all keys from all shards (parse all shards in the train split).

        Args:
          seen_keys: Set of all keys that are present in the test split.
          dpath: Root path of this dataset.
          train_shards: Description of train shards
            (self.shards_list[Dataset.TRAIN_SPLIT]).
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
          examples_per_shard: Number of examples in each shard.
          key_ap: The attack point that is checked for when checking
            disjointness of splits.
        """
        for i in pbar(range(len(train_shards)),
                      desc="Checking test key uniqueness"):
            for example in Dataset.inspect(dataset_path=dpath,
                                           split=Dataset.TRAIN_SPLIT,
                                           shard_id=i,
                                           num_example=examples_per_shard,
                                           verbose=False).as_numpy_iterator():
                cur_key = example[key_ap].astype(np.uint8).tobytes()
                if cur_key in seen_keys:
                    raise ValueError(
                        f"Duplicate key: {cur_key!r} in test split, in "
                        f"{train_shards[i]}")

    def get_config_dictionary(self) -> Dict[str, Any]:
        """Return dictionary of information about this dataset.

        Raises: ValueError if saving this dictionary using json would cause
          data loss. This can be caused by having different keys with the same
          string representation:

          d = {0: 1, "0": 2}  # JSON key collision
          l = json.loads(json.dumps(d))
          assert l != d

          Note that it is ok to have keys of other type than string, since the
          check is performed using Dataset._from_loaded_json.
        """
        representation = {
            "shortname": self.shortname,
            "architecture": self.architecture,
            "implementation": self.implementation,
            "algorithm": self.algorithm,
            "version": self.version,
            "firmware_sha256": self.firmware_sha256,
            "url": self.url,
            "firmware_url": self.firmware_url,
            "paper_url": self.paper_url,
            "licence": self.licence,
            "description": self.description,
            "compression": self.compression,
            "shards_list": self.shards_list,
            "keys_per_group": self.keys_per_group,
            "keys_per_split": self.keys_per_split,
            "examples_per_group": self.examples_per_group,
            "examples_per_shard": self.examples_per_shard,
            "examples_per_split": self.examples_per_split,
            "capture_info": self.capture_info,
            "measurements_info": self.measurements_info,
            "attack_points_info": self.attack_points_info,
            "min_values": self.min_values,
            "max_values": self.max_values,
            # See scaaml.__version__ docstring for more information.
            "scaaml_version": scaaml.__version__,
            "measurement_dtype": dtype_dtype_to_name(self.measurement_dtype),
        }
        loaded = Dataset._from_loaded_json(
            json.loads(json.dumps(representation)))
        if loaded != representation:
            pprint_file = self.path / f"info.{time()}.pprint"
            pprint_file.write_text(pprint.pformat(representation))
            raise ValueError(f"JSON representation causes data loss, saving "
                             f"into {pprint_file}")
        return representation

    @staticmethod
    def _load_config(conf_path: Path) -> Dict[str, Any]:
        """Get config dictionary from a file. Use this function instead of an
        json.loads, as this function returns correct types for group ids.

        Args:
          conf_path: Path object representing the dataset information (e.g.,
            the return value of Dataset._get_config_path).

        Returns: Dictionary representation of the Dataset.
        """
        return Dataset._from_loaded_json(json.loads(conf_path.read_text()))

    @staticmethod
    def _from_loaded_json(loaded_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix types in the data-structure loaded from JSON. Necessary as JSON
        allows only string keys, but for instance group keys are integers in
        Dataset.

        Args:
          loaded_dict: The data-structure returned by json.load on the
            info.json file.

        Returns: The same information with fixed types.
        """
        fixed_dict = copy.deepcopy(loaded_dict)
        find_misspellings(fixed_dict.keys())  # Check for misspellings of keys.
        # Fix type of keys_per_group
        fixed_dict["keys_per_group"] = {
            split: {
                int(group): n_examples
                for group, n_examples in keys_info.items()
            } for split, keys_info in loaded_dict["keys_per_group"].items()
        }
        # Fix type of examples_per_group
        fixed_dict["examples_per_group"] = {
            split: {
                int(group): n_examples for group, n_examples in ex_info.items()
            } for split, ex_info in loaded_dict["examples_per_group"].items()
        }
        # Fix missing keys
        if "licence" not in fixed_dict:
            # Do not re-licence
            fixed_dict["licence"] = ""
        for k in ["firmware_url", "paper_url"]:
            if k not in fixed_dict:
                fixed_dict[k] = ""
        return fixed_dict

    def _write_config(self) -> None:
        """Save configuration as json."""
        with open(self._get_config_path(self.path), "w+",
                  encoding="utf-8") as f:
            json.dump(self.get_config_dictionary(), f)

    def write_config(self) -> None:
        """Force updating `info.json` file."""
        self._write_config()

    @staticmethod
    def from_config(dataset_path: str, verbose: bool = True) -> "Dataset":
        """Load a dataset from a config file.

        Args:
          dataset_path: The path to the dataset.
          verbose: Print config path and dataset path.

        Raises: ValueError if the dataset version is higher than the scaaml
          module used. See scaaml.__version__ docstring.
        """
        dpath = Path(dataset_path)
        conf_path = Dataset._get_config_path(dataset_path)
        if verbose:
            cprint(f"reloading {conf_path}", "magenta")
        config = Dataset._load_config(conf_path)
        # Check that the library version (version of this software) is not
        # lower than what was used to capture the dataset.
        if "scaaml_version" in config.keys():
            if semver.Version.parse(config["scaaml_version"]).compare(
                    scaaml.__version__) > 0:
                version_in_config = config["scaaml_version"]
                raise ValueError(f"SCAAML module is outdated, scaaml_version: "
                                 f"{scaaml.__version__}, but dataset was "
                                 f"created using: {version_in_config}")
        return Dataset(
            root_path=str(dpath),
            shortname=config["shortname"],
            architecture=config["architecture"],
            implementation=config["implementation"],
            algorithm=config["algorithm"],
            version=config["version"],
            url=config["url"],
            firmware_url=config["firmware_url"],
            paper_url=config["paper_url"],
            licence=config["licence"],
            description=config["description"],
            firmware_sha256=config["firmware_sha256"],
            measurements_info=config["measurements_info"],
            attack_points_info=config["attack_points_info"],
            capture_info=config["capture_info"],
            compression=config["compression"],
            shards_list=config["shards_list"],
            keys_per_group=config["keys_per_group"],
            keys_per_split=config["keys_per_split"],
            examples_per_group=config["examples_per_group"],
            examples_per_split=config["examples_per_split"],
            examples_per_shard=config["examples_per_shard"],
            min_values=config["min_values"],
            max_values=config["max_values"],
            from_config=True,
            verbose=verbose,
            measurement_dtype=dtype_name_to_dtype(
                config.get("measurement_dtype", "float32")),
        )

    @staticmethod
    def _get_config_path(path: Union[str, Path]) -> Path:
        return Path(path) / "info.json"

    @staticmethod
    def _cleanup_shards(dataset_path: Path,
                        print_info: bool = True) -> Dict[str, Any]:
        """Returns an updated config which contains only shards that correspond
        to existing files.

        Args:
          dataset_path: The directory of the dataset. It is assumed that there
            is a file dataset_path/'info.json' (see Dataset._get_config_path)
            with the configuration in JSON format and subdirectories train,
            test, holdout.
          print_info: Print how many shards were removed and how many were kept.

        Returns: Updated configuration to be written using json.dump.
        """
        config = Dataset._load_config(Dataset._get_config_path(dataset_path))
        stats = []  # Statistic how many were kept and removed in each split.
        new_shards_list = defaultdict(list)
        for split, shard_list in config["shards_list"].items():
            kept = 0
            removed = 0
            for shard in shard_list:
                shard_path = dataset_path / shard["path"]  # The shard.
                if shard_path.exists():
                    new_shards_list[split].append(shard)
                    kept += 1
                else:
                    removed += 1
            assert kept + removed == len(shard_list)
            assert len(new_shards_list[split]) == kept
            stats.append([split, kept, removed])

        config["shards_list"] = new_shards_list
        examples_per_split = {}
        examples_per_group: Dict[str, Dict[int, int]] = {}
        for split, shard_list in config["shards_list"].items():
            examples_per_split[split] = config["examples_per_shard"] * len(
                config["shards_list"][split])
            # Zero examples per a group is a valid option.
            examples_per_group[split] = {
                k: 0 for k in config["examples_per_group"][split]
            }
            names_keys_per_group: Dict[int, Set[Any]] = {
                k: set() for k in config["keys_per_group"][split]
            }
            key_names_per_split = set()
            for shard in shard_list:
                shard_info = Dataset._shard_info_from_name(shard["path"])
                sg: int = int(shard_info["shard_group"])
                examples_per_group[split][sg] += config["examples_per_shard"]
                key_names_per_split.add(shard_info["shard_key"])
                names_keys_per_group[sg].add(shard_info["shard_key"])
            # We suppose that each shard contains at most one key.
            config["keys_per_group"][split] = {
                k: len(names_keys_per_group[k])
                for k in config["keys_per_group"][split]
            }
            config["keys_per_split"][split] = len(key_names_per_split)

        config["examples_per_split"] = examples_per_split
        config["examples_per_group"] = examples_per_group
        if print_info:
            print(tabulate(stats, headers=["split", "kept", "removed"]))
        return config

    @staticmethod
    def cleanup_shards(dataset_path: Union[str, Path]) -> None:
        """Remove non_existing shards from the config and update the config.
        Makes a backup of the old config named info.json.sav.{time()}.json.

        Args:
          dataset_path: The directory of the dataset.

        Example use:
          Dataset.cleanup_shards(
              dataset_path="/mnt/storage/chipwhisperer/test_cleanup")
        """
        fpath = Dataset._get_config_path(dataset_path)

        # Save the old config.
        save_path = Path(f"{str(fpath)}.sav.{time()}.json")
        assert not save_path.exists()  # Do not overwrite.
        save_path.write_text(fpath.read_text(encoding="utf-8"),
                             encoding="utf-8")
        cprint(f"Saving old config to {save_path}", "cyan")

        # Rewrite with the new config.
        cprint("Writing cleaned config", "green")
        new_config = Dataset._cleanup_shards(dataset_path=Path(dataset_path),
                                             print_info=True)
        with open(fpath, "w+", encoding="utf-8") as o:
            json.dump(new_config, o)

    def _move_shard(self, from_split: str, to_split: str,
                    shard_idx: int) -> None:
        """Move a single shard. If this method raises the dataset might be left
        in a state which is not valid.

        The user must update keys_per_split and keys_per_group values (this
        should be done in move_shards). This is needed as single key may be used
        in different shards.

        Args:
          from_split: The split that loses a shard.
          to_split: The split that gains a shard.
          shard_idx: The index to the shards. Valid values are in
            range(len(shards_list[from_split])).

        Raises:
          IndexError: If shard_idx is out of bounds.
        """
        if not shard_idx in range(len(self.shards_list[from_split])):
            raise IndexError(f"0 <= shard_idx < len(shards_list[from_split]) "
                             f"violated: 0 <= {shard_idx} < "
                             f"{len(self.shards_list[from_split])}")
        # The shard to move.
        shard = self.shards_list[from_split][shard_idx]
        # Move the file.
        shard_file = self.path / shard["path"]
        new_name = Dataset._shard_name(shard_group=shard["group"],
                                       shard_key=shard["key"],
                                       shard_part=shard["part"])
        shard["path"] = f"{to_split}/{new_name}"
        moved_file = self.path / shard["path"]
        # Make sure the directory exists (in case the dataset was downloaded
        # it can happen that empty directory is not created).
        moved_file.parent.mkdir(exist_ok=True)
        # Move the shard file.
        shard_file.rename(moved_file)
        # Move the shard object.
        if to_split not in self.shards_list.keys():
            self.shards_list[to_split] = [shard]
        else:
            self.shards_list[to_split].insert(0, shard)
        del self.shards_list[from_split][shard_idx]
        # Fix metadata.
        self.examples_per_split[from_split] -= shard["examples"]
        self.examples_per_split[to_split] += shard["examples"]
        tmp = cast(Dict[str, int], self.examples_per_group[from_split])
        tmp[shard["group"]] -= shard["examples"]
        # Zero value should not be present.
        tmp = cast(Dict[str, int], self.examples_per_group[from_split])
        if tmp[shard["group"]] == 0:
            del tmp[shard["group"]]
        tmp = cast(Dict[str, int], self.examples_per_group[to_split])
        tmp[shard["group"]] += shard["examples"]

    def move_shards(self, from_split: str, to_split: str,
                    shards: Union[int, Set[int]]) -> None:
        """Move shards from one split to another. This method modifies this
        dataset (moves files). Make a backup before calling this method.

        Args:
          from_split: The split that loses shards.
          to_split: The split that gains shards.
          shards: If this parameter is an integer, then the first shards from
            shards_list are moved. If shards is a set of integers then the
            shards of the corresponding indices are moved.

        Raises:
          ValueError: If the check method fails.

        Example use:
          # Make a backup of the original dataset.
          # Either call:
          dataset.move_shards(from_split=Dataset.TRAIN_SPLIT,
                              to_split=Dataset.TEST_SPLIT,
                              shards=3)
          # Or call:
          dataset.move_shards(from_split=Dataset.TRAIN_SPLIT,
                              to_split=Dataset.TEST_SPLIT,
                              shards={0, 1, 2})
        """
        if isinstance(shards, int):
            shards = set(range(shards))
        keys_moved = defaultdict(set)
        # Move the shards one by one in reversed order (to keep all indexes
        # valid).
        for shard_idx in sorted(shards, reverse=True):
            shard = self.shards_list[from_split][shard_idx]
            # Remember moving this key.
            keys_moved[shard["group"]].add(shard["key"])
            self._move_shard(from_split=from_split,
                             to_split=to_split,
                             shard_idx=shard_idx)
        # Update keys_per_split and keys_per_group.
        all_keys = set()
        for group_keys in keys_moved.values():
            all_keys.update(group_keys)
        self.keys_per_split[from_split] -= len(all_keys)
        self.keys_per_split[to_split] += len(all_keys)
        for group in self.keys_per_group[
                from_split]:  # type: ignore[attr-defined]
            tmp = cast(Dict[Any, int], self.keys_per_group[from_split])
            tmp[group] -= len(keys_moved[group])
            tmp = cast(Dict[Any, int], self.keys_per_group[to_split])
            tmp[group] += len(keys_moved[group])
        # Write config to save the state of this dataset. (Also the config is
        # used in self.check()).
        self._write_config()
        # Check the resulting dataset (especially for key repetition).
        self.check()

    def merge_with(self, other_dataset: "Dataset") -> None:
        """Merge other_dataset into this dataset. This method changes this
        dataset (self). Make a backup before calling this method.

        This method assumes that other_dataset contains no key that is also
        present in this dataset.

        Args:
          other_dataset: Another dataset object to copy shards from. Does not
            get changed. Should be of the same type (same firmware_sha256,
            compression, examples_per_shard, measurements_info, licence, ....).

        The following properties are not updated (and not checked to be equal
        to those of other_dataset):
          shortname
          version
          description
          url
          Current shard tracking: shard_key, prev_shard_key, shard_path,
            shard_split, shard_part, shard_relative_path, curr_shard

        The following are updated by the merge:
          shards_list
          keys_per_group
          keys_per_split
          examples_per_group
          examples_per_split
          examples_per_shard
          min_values
          max_values

        Raises:
          ValueError: If Dataset.check fails.
          FileExistsError: If a shard should be copied over an existing file.
            Assume that after this error self is not in a consistent state.
        """
        # The following properties must be the same in order for merge to make
        # sense:
        assert self.firmware_sha256 == other_dataset.firmware_sha256
        #assert self.firmware_url == other_dataset.firmware_url,
        #assert self.paper_url == other_dataset.paper_url,
        #assert self.licence == other_dataset.licence
        assert self.architecture == other_dataset.architecture
        assert self.implementation == other_dataset.implementation
        assert self.algorithm == other_dataset.algorithm
        assert self.compression == other_dataset.compression
        assert self.capture_info == other_dataset.capture_info
        assert self.measurements_info == other_dataset.measurements_info
        assert self.attack_points_info == other_dataset.attack_points_info
        assert self.min_values.keys() == other_dataset.min_values.keys()
        assert self.max_values.keys() == other_dataset.max_values.keys()
        assert self.examples_per_shard == other_dataset.examples_per_shard

        # Update meta-information.
        # Update extreme values:
        self.min_values = {
            k: min(v, other_dataset.min_values[k])
            for k, v in self.min_values.items()
        }
        self.max_values = {
            k: max(v, other_dataset.max_values[k])
            for k, v in self.max_values.items()
        }

        # Update shards.
        for split in other_dataset.shards_list:
            seen_keys: Set[Any] = set()
            seen_keys_per_group: Dict[Any, Any] = {}
            for shard in tqdm(other_dataset.shards_list[split],
                              desc=f"Merging {split}"):
                self.shards_list[split].append(shard)
                seen_keys.add(shard["key"])
                if shard["group"] not in seen_keys_per_group:
                    seen_keys_per_group[shard["group"]] = {shard["key"]}
                else:
                    seen_keys_per_group[shard["group"]].add(shard["key"])

                # Copy the file.
                other_file = other_dataset.path / shard["path"]
                copied_file = self.path / shard["path"]
                if copied_file.exists():
                    raise FileExistsError(f"Shard file {copied_file} already "
                                          f"exists.")
                # Python3.7 shutil.copy takes string arguments.
                shutil.copy(str(other_file), str(copied_file))

                # Update metadata.
                tmp = cast(Dict[Any, int], self.examples_per_group[split])
                tmp[shard["group"]] += shard["examples"]
                self.examples_per_split[split] += shard["examples"]

            self.keys_per_split[split] += len(seen_keys)
            for group, key_set in seen_keys_per_group.items():
                tmp = cast(Dict[Any, int], self.keys_per_group[split])
                tmp[group] += len(key_set)

        # Write config to save the state of this dataset. (Also the config is
        # used in self.check()).
        self._write_config()
        # Check the resulting dataset.
        self.check()
