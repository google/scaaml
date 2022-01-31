"""Build and load tensorFlow dataset Record wrapper"""

import copy
import math
import json
import os
from collections import defaultdict
from time import time
from typing import Dict, List, Optional, Union
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
from .shard import Shard
from .errors import DatasetExistsError


class Dataset():
    def __init__(
        self,
        root_path: str,
        shortname: str,
        architecture: str,
        implementation: str,
        algorithm: str,
        version: int,
        firmware_sha256: str,
        description: str,
        examples_per_shard: int,
        measurements_info: Dict,
        attack_points_info: Dict,
        url: str,
        firmware_url: str = '',
        paper_url: str = '',
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        compression: str = "GZIP",
        shards_list: Optional[Dict[str, List]] = None,
        keys_per_group: Optional[Dict[str, Dict[int, int]]] = None,
        keys_per_split: Optional[Dict[str, int]] = None,
        examples_per_group: Optional[Dict[str, Dict[int, int]]] = None,
        examples_per_split: Optional[Dict[str, int]] = None,
        capture_info: Optional[dict] = None,
        min_values: Optional[Dict[str, int]] = None,
        max_values: Optional[Dict[str, int]] = None,
        from_config: bool = False,
    ) -> None:
        """Class for saving and loading a database.

        Args:
          url: Where to download this dataset.
          firmware_url: Where to dowload the firmware used while capture.
          paper_url: Where to find the published paper.
          licence: URL or the whole licence the dataset is published under.
          from_config: This Dataset object has been created from a saved
            config, root_path thus points to what should be self.path. When
            True set self.path = root_path, self.root_path to be the parent of
            self.path. In this case it does not necessarily hold that
            self.path.name == self.slug (the directory could have been renamed).

        Raises:
          ValueError: If firmware_sha256 or firmware_url evaluates to False.
          DatasetExistsError: If creating this object would overwrite the
            corresponding config file.
        """
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

        if not self.firmware_sha256:
            raise ValueError("Firmware hash is required")
        if not self.firmware_url:
            raise ValueError("Firmware URL is required")

        self.slug = "%s_%s_%s_v%s_%s" % (shortname, algorithm, architecture,
                                         implementation, version)
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
                Path(self.path / 'train').mkdir()
                Path(self.path / 'test').mkdir()
                Path(self.path / 'holdout').mkdir()

        cprint("Dataset path: %s" % self.path, 'green')

        # current shard tracking
        self.shard_key = None
        self.prev_shard_key = None  # track key change for counting
        self.shard_path = None
        self.shard_split = None
        self.shard_part = None
        self.shard_relative_path = None  # for the shardlist
        self.curr_shard = None  # current_ shard object

        # [counters] - must be passed as param to allow reload.
        # shards_list[split] is a list of shard info dictionaries (where split
        # in ['test', 'train', 'holdout']
        self.shards_list = siutils.ddict(value=shards_list,
                                         levels=1,
                                         type_var=list)

        # keys counting
        # keys_per_group[split][group_id] contains the number (int) of keys
        # belonging to the group (group_id is int)
        self.keys_per_group = siutils.ddict(value=keys_per_group,
                                            levels=2,
                                            type_var=int)
        self.keys_per_split = siutils.ddict(value=keys_per_split,
                                            levels=1,
                                            type_var=int)

        # examples counting
        # keys_per_group[split][gid] = cnt
        self.examples_per_group = siutils.ddict(value=examples_per_group,
                                                levels=2,
                                                type_var=int)
        self.examples_per_split = siutils.ddict(value=examples_per_split,
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
                self.max_values[k] = 0

        # write config if needed
        if not from_config:
            self._write_config()

    @staticmethod
    def get_dataset(*args, **kwargs):
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
            return Dataset.from_config(dataset_path=err.dataset_path)

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
        sname = "%s_%s_%s.tfrec" % (shard_group, shard_key, shard_part)
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
        for dir_separator in ['\\', '/']:
            if dir_separator in shard_name:
                shard_name = shard_name.split(dir_separator)[-1]
        parts = shard_name.split('_')
        kwargs = {}
        kwargs['shard_group'] = int(parts[0])
        kwargs['shard_key'] = parts[1]
        kwargs['shard_part'] = int(parts[2].split('.')[0])
        return kwargs

    def new_shard(self, key: list, part: int, group: int, split: str,
                  chip_id: int):
        """Initiate a new key

        Args:
            key: the key that was used to create the measurements.

            part: Indicate which part of a given key set of catpure this
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

        if split not in ['train', 'test', 'holdout']:
            raise ValueError("Invalid split, must be: {train, test, holdout}")

        if part < 0 or part > 10:
            raise ValueError("Invalid part value -- muse be in [0, 10]")

        self.shard_split = split
        self.shard_part = part
        self.shard_group = group
        self.shard_key = bytelist_to_hex(key, spacer='')
        self.shard_chip_id = chip_id

        # shard name
        fname = Dataset._shard_name(self.shard_group, self.shard_key,
                                    self.shard_part)
        self.shard_relative_path = "%s/%s" % (split, fname)
        self.shard_path = str(self.path / self.shard_relative_path)

        # new shard
        self.curr_shard = Shard(self.shard_path,
                                attack_points_info=self.attack_points_info,
                                measurements_info=self.measurements_info,
                                compression=self.compression)

    def write_example(self, attack_points: Dict, measurement: Dict):
        self.curr_shard.write(attack_points, measurement)

    def close_shard(self):
        # close the shard
        stats = self.curr_shard.close()
        if stats['examples'] != self.examples_per_shard:
            cprint(
                f"This shard contains {stats['examples']}, expected "
                f"{self.examples_per_shard}", 'red')

        # update min/max values
        for k, v in stats['min_values'].items():
            self.min_values[k] = min(self.min_values[k], v)

        for k, v in stats['max_values'].items():
            self.max_values[k] = max(self.max_values[k], v)

        # update key stats only if key changed
        if self.shard_key != self.prev_shard_key:
            self.keys_per_split[self.shard_split] += 1
            self.keys_per_group[self.shard_split][self.shard_group] += 1
            self.prev_shard_key = self.shard_key

        self.examples_per_split[self.shard_split] += stats['examples']
        self.examples_per_group[self.shard_split][
            self.shard_group] += stats['examples']

        # record in shardlist
        self.shards_list[self.shard_split].append({
            "path": str(self.shard_relative_path),
            "examples": stats['examples'],
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
    def download(url: str):
        "Download dataset from a given url"
        raise NotImplementedError("implement me using keras dl mechanism")

    @staticmethod
    def as_tfdataset(dataset_path: str,
                     split: str,
                     attack_points: Union[List[str], str],
                     traces: Union[List[str], str],
                     bytes: Union[List, int],
                     shards: int = None,
                     parts: Union[List[int], int] = None,
                     trace_start: int = 0,
                     trace_len: int = None,
                     batch_size: int = 32,
                     prefetch: int = tf.data.AUTOTUNE,
                     file_parallelism: int = os.cpu_count(),
                     parallelism: int = tf.data.AUTOTUNE,
                     shuffle: int = 1000
                     ) -> Union[tf.data.Dataset, Dict, Dict]:
        """"Dataset as tfdataset

        FIXME: restrict shards to specific part if they exists.

        """

        if parts:
            raise NotImplementedError("Implement part filtering")


        # boxing
        if isinstance(traces, str):
            traces = [traces]
        if isinstance(bytes, int):
            bytes = [bytes]
        if isinstance(attack_points, str):
            attack_points = [attack_points]

        # loading info
        dpath = Path(dataset_path)
        dataset = Dataset.from_config(dataset_path)

        if split not in dataset.keys_per_split:
            raise ValueError("Unknown split -- see Dataset.summary() for list")

        # TF_FEATURES construction: must contains all features and be global
        tf_features = {}  # what is decoded
        for name, ipt in dataset.measurements_info.items():
            tf_features[name] = tf.io.FixedLenFeature([ipt['len']], tf.float32)
        for name, ap in dataset.attack_points_info.items():
            tf_features[name] = tf.io.FixedLenFeature([ap['len']], tf.int64)

        # decoding funtion
        def from_tfrecord(tfrecord):
            rec = tf.io.parse_single_example(tfrecord, tf_features)
            return rec

        # inputs construction
        inputs = {}  # model inputs
        for name in traces:
            ipt = dataset.measurements_info[name]
            inputs[name] = ipt

            inputs[name]['min'] = tf.constant(dataset.min_values[name])
            inputs[name]['max'] = tf.constant(dataset.max_values[name])
            delta = tf.constant(inputs[name]['max'] - inputs[name]['min'])
            inputs[name]['delta'] = delta


        # output construction
        outputs = {}  # model outputs
        for name in attack_points:
            for b in bytes:
                n = "%s_%s" % (name, b)
                ap = dataset.attack_points_info[name]
                outputs[n] = ap
                outputs[n]['ap'] = name
                outputs[n]['byte'] = b

        # processing function
        # @tf.function
        def process_record(rec):
            "process the tf record to get it ready for learning"
            x = {}
            # normalize the traces
            for name, data in inputs.items():
                trace = rec[name]

                # truncate if needed
                if trace_start:
                    trace = trace[trace_start:]

                if trace_len:
                    trace = trace[:trace_len]

                # rescale
                # trace = 2 * ((trace - data['min']) / (data['delta'])) - 1

                # reshape
                # trace = tf.reshape(trace, (reshaped_trace_len, step_size))

                # assign
                x[name] = trace
                inputs[name]['shape'] = trace.shape  # (trace_len - trace_start)
            # one_hot the outptut for each ap/byte
            y = {}
            for name, data in outputs.items():
                v = tf.one_hot(rec[data['ap']][data['byte']], data['max_val'])
                y[name] = v

            return (x, y)

        # collect and truncate shard list of a given split
        # this is done prior to anything to allow to only download the nth
        # first shards
        shards_list = dataset.shards_list[split]
        if shards:
            shards_list = shards_list[:shards]
        shards_paths = [str(dpath / s['path']) for s in shards_list]
        num_shards = len(shards_paths)
        # print(shards_paths)
        # dataset creation
        # with tf.device('/cpu:0'):
        # shuffle the shard order
        ds = tf.data.Dataset.from_tensor_slices(shards_paths)
        ds = ds.repeat()
        # shuffle shard order
        ds = ds.shuffle(num_shards)

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the faster and more compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=dataset.compression),  # noqa
            cycle_length=file_parallelism,
            block_length=1,
            num_parallel_calls=file_parallelism,
            deterministic=False)
        # decode to records
        ds = ds.map(from_tfrecord, num_parallel_calls=parallelism)
        # process them
        ds = ds.map(process_record, num_parallel_calls=parallelism)

        # # randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            ds = ds.shuffle(shuffle)

        # # batching with repeat
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch)

        return ds, inputs, outputs

    @staticmethod
    def summary(dataset_path):
        """Print a summary of the dataset"""
        lst = ['shortname',
               'description', 'url',
               'architecture', 'implementation',
               'algorithm', 'version', 'compression']

        conf_path = Dataset._get_config_path(dataset_path)
        config = Dataset._load_config(conf_path)
        cprint("[Dataset Summary]", 'cyan')
        cprint("Info", 'yellow')
        print(tabulate([[k, config.get(k, '')] for k in lst]))

        cprint("\nAttack Points", 'yellow')
        d = [[k, v['len'], v['max_val']]
             for k, v in config['attack_points_info'].items()]
        print(tabulate(d, headers=['ap', 'len', 'max_val']))

        cprint("\nMeasurements", 'magenta')
        d = [[k, v['type'], v['len']]
             for k, v in config['measurements_info'].items()]
        print(tabulate(d, headers=['name', 'type', 'len']))

        cprint("\nContent", 'green')
        d = []
        for split in config['keys_per_split'].keys():
            d.append([
                split,
                len(config['shards_list'][split]),
                config['keys_per_split'][split],
                config['examples_per_split'][split],
            ])
        print(tabulate(d, ['split', 'num_shards', 'num_keys', 'num_examples']))

    @staticmethod
    def inspect(dataset_path,
                split: str,  # typing.Literal['train', 'test', 'holdout'],
                shard_id: int,
                num_example: int,
                verbose: bool = True):
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
        shard_path = Path(
            dataset_path) / config['shards_list'][split][shard_id]['path']
        if verbose:
            cprint(f'Reading shard {shard_path}', 'cyan')
        s = Shard(str(shard_path),
                  attack_points_info=config['attack_points_info'],
                  measurements_info=config['measurements_info'],
                  compression=config['compression'])
        data = s.read(num=num_example)
        if verbose:
            print(data)
        return data

    def check(self, deep_check: bool = True, show_progressbar: bool = True):
        """Check the dataset integrity. Check integrity of metadata in config
        and also that no key from the train is in the test.

        Args:
          deep_check: When checking that keys in test and train splits are
            disjoint inspect train shards (set to True if a single train shard
            may contain multiple different keys).
          show_progressbar: Use tqdm to show a progressbar for different checks.

        Raises: ValueError if the dataset is inconsistent.
        """
        if show_progressbar:
            pbar = tqdm
        else:
            # Redefine tqdm to the identity function returning the first unnamed
            # parameter.
            pbar = lambda *args, **kwargs: args[0]

        Dataset._check_metadata(config=self._get_config_dictionary())
        Dataset._check_sha256sums(shards_list=self.shards_list,
                                  dpath=Path(self.path),
                                  pbar=pbar)
        # Check shard metadata
        for slist in self.shards_list.values():
            for shard_info in slist:
                Dataset._check_shard_metadata(shard_info=shard_info,
                                              dataset_path=self.path)
        # Ensure that no keys in the train split are present in the test split.
        if 'test' in self.examples_per_split and 'train' in self.examples_per_split:
            self._check_disjoint_keys(pbar=pbar, deep_check=deep_check)

    def _check_disjoint_keys(self, pbar, deep_check: bool = True):
        """Check that no key in the train split is present in the test split.

        Args:
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
          deep_check: When checking that keys in test and train splits are
            disjoint inspect train shards (set to True if a single train shard
            may contain multiple different keys).

        Raises: ValueError if some key from train is present in test.
        """
        seen_keys = set()
        for i in range(len(self.shards_list['test'])):
            for example in Dataset.inspect(
                    dataset_path=self.path,
                    split='test',
                    shard_id=i,
                    num_example=self.examples_per_shard,
                    verbose=False).as_numpy_iterator():
                seen_keys.add(example['key'].astype(np.uint8).tobytes())
        if deep_check:
            Dataset._deep_check(seen_keys=seen_keys,
                                dpath=self.path,
                                train_shards=self.shards_list['train'],
                                pbar=pbar,
                                examples_per_shard=self.examples_per_shard)
        else:
            Dataset._shallow_check(seen_keys=seen_keys,
                                   train_shards=self.shards_list['train'],
                                   pbar=pbar)

    @staticmethod
    def _check_sha256sums(shards_list, dpath: Path, pbar):
        """Check the metadata of this dataset.

        Args:
          shards_list: Dictionary with information about each shard.
            Use _get_config_dictionary()['shards_list']
          dpath: Root path of the dataset.
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).

        Raises: ValueError if some hash does not match.
        """
        for split, slist in shards_list.items():
            for sinfo in pbar(slist, desc=f'Checking sha for {split}'):
                shard_path = dpath / sinfo['path']
                sha_hash = siutils.sha256sum(shard_path)
                if sha_hash != sinfo['sha256']:
                    raise ValueError(sinfo['path'], "SHA256 miss-match")

    @staticmethod
    def _check_shard_metadata(shard_info: Dict, dataset_path: Path) -> None:
        """Checks shard metadata.

        Args:
          shard_info: Dictionary of the shard metadata.
          dataset_path: Dataset path, so that we can check size of the shard
            file.

        Raises: ValueError if the metadata is inconsistent.
        """
        # Check that only expected keys are present:
        si_keys = {
            'examples',  # Checked by Dataset._check_metadata
            'sha256',  # Checked by Dataset._check_sha256sums
            'path',  # Checked by Dataset._check_sha256sums
            'group',  # Checked against path
            'key',  # Checked against path
            'part',  # Checked against path
            'size',  # Checked here
            'chip_id',  # Checked that it is a non-negative integer
        }
        if set(shard_info.keys()) != si_keys:
            raise ValueError(f'Shard info keys are: {shard_info.keys()} '
                             f'expected: {si_keys}, in shard: {shard_info}')
        # Check that the info corresponds to the filename:
        file_info = Dataset._shard_info_from_name(shard_info['path'])
        for key in ['group', 'part']:
            if file_info['shard_' + key] != shard_info[key]:
                raise ValueError(f'{key} does not match filename, expected: '
                                 f'{file_info["shard_" + key]}, got: '
                                 f'{shard_info[key]}, in shard: {shard_info}')
        # Check key (in filename it is lower case, in info it is upper case)
        if file_info['shard_key'].lower() != shard_info['key'].lower():
            raise ValueError(f'key does not match filename, expected: '
                             f'{file_info["shard_key"]}, got: '
                             f'{shard_info["key"]} (not case sensitive), in '
                             f'shard: {shard_info}')
        # Check size of the file
        size = os.stat(dataset_path / shard_info['path']).st_size
        if size != shard_info['size']:
            raise ValueError(f'Wrong size, got: {size}, expected: '
                             f'{shard_info["size"]}, in shard: {shard_info}')
        # Check chip_id is non-negative integer
        chip_id = shard_info['chip_id']
        if not isinstance(chip_id, int) or chip_id < 0:
            raise ValueError(f'Wrong chip_id, got: {chip_id}, of type: '
                             f'{type(chip_id)}, in shard: {shard_info}')

    @staticmethod
    def _check_metadata(config,
                        n_examples_in_each_shard_is_constant: bool = False):
        """Check the metadata of this dataset.

        Args:
          config: A dictionary representing the metadata.
          n_examples_in_each_shard_is_constant: Check that each shard contains
            exactly examples_per_shard examples.

        Raises: ValueError if some metadata do not match.
        """
        for split, expected_examples in config['examples_per_split'].items():
            slist = config['shards_list'][split]
            # checking we have the rigt number of shards
            if len(slist) != expected_examples // config['examples_per_shard']:
                raise ValueError("Num shards in shard_list != "
                                 "examples_per_split // examples_per_shard")
            # Check that expected_examples is a multiple of examples_per_shard.
            if expected_examples % config['examples_per_shard']:
                raise ValueError("expected_examples is not divisible by "
                                 "examples_per_shard")

            if expected_examples != sum(s['examples'] for s in slist):
                raise ValueError(f'Mismatch in expected_examples, shards '
                                 f'metadata do not agree in {split}.')
            if n_examples_in_each_shard_is_constant:
                # All shards have the same number of examples.
                if any(s['examples'] != config['examples_per_shard']
                       for s in slist):
                    raise ValueError(f'Not all shards in {split} contain the '
                                     f'same number of examples.')

            # Check examples_per_group sums to the right thing.
            sum_examples_per_group = sum(
                config['examples_per_group'][split].values())
            if sum_examples_per_group != expected_examples:
                raise ValueError(f'Wrong sum of examples_per_group in {split}')
            # Check examples_per_group in individual groups.
            # Dataset.check can be called either after creating a dataset (when
            # all measurements are done) or after loading from a config. The
            # JSON file-format only allows keys to be strings. When the dataset
            # is created the group ids are integers, but when dataset is loaded
            # they are strings. We check the case where all keys are strings.
            examples_per_group = defaultdict(int)
            for shard in slist:
                examples_per_group[str(shard['group'])] += shard['examples']
            examples_per_group_config = {
                str(k): v
                for k, v in config['examples_per_group'][split].items()
            }
            if examples_per_group != examples_per_group_config:
                raise ValueError(f"Wrong examples_per_group in {split}")

            actual_examples = 0
            for sinfo in slist:
                actual_examples += sinfo['examples']
                if sinfo['examples'] != config['examples_per_shard']:
                    raise ValueError(f'Wrong number of examples, expected: '
                                     f'{config["examples_per_shard"]}, got: '
                                     f'{sinfo["examples"]}, in shard: {sinfo}')

            if actual_examples != expected_examples:
                raise ValueError("sum example don't match top_examples")

    @staticmethod
    def _shallow_check(seen_keys, train_shards, pbar):
        """Check just what is in self.shards_list info (do not parse all
        shards).

        Args:
          seen_keys: Set of all keys that are present in the test split.
          train_shards: Description of train shards (self.shards_list['train']).
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
        """
        for shard in pbar(train_shards, desc='Checking test key uniqueness'):
            k = shard['key'].lower()
            list_k = [int(k[2 * i:2 * i + 2], 16) for i in range(len(k) // 2)]
            cur_key = np.array(list_k, dtype=np.uint8).tobytes()
            if cur_key in seen_keys:
                raise ValueError(
                    f'Duplicate key: {k} in test split, in {shard}')

    @staticmethod
    def _deep_check(seen_keys, dpath, train_shards, pbar,
                    examples_per_shard: int):
        """Check all keys from all shards (parse all shards in the train split).

        Args:
          seen_keys: Set of all keys that are present in the test split.
          dpath: Root path of this dataset.
          train_shards: Description of train shards (self.shards_list['train']).
          pbar: Either tqdm.tqdm or an identity function (in order not to
            print).
          examples_per_shard: Number of examples in each shard.
        """
        for i in pbar(range(len(train_shards)),
                      desc='Checking test key uniqueness'):
            for j, example in enumerate(
                    Dataset.inspect(
                        dataset_path=dpath,
                        split='train',
                        shard_id=i,
                        num_example=examples_per_shard,
                        verbose=False).as_numpy_iterator()):
                cur_key = example['key'].astype(np.uint8).tobytes()
                if cur_key in seen_keys:
                    raise ValueError(
                        f'Duplicate key: {cur_key} in test split, in '
                        f'{train_shards[i]}')

    def _get_config_dictionary(self):
        """Return dictionary of information about this dataset.

        Raises: ValueError if saving this dictionary using json would cause
          data loss. This can be caused by having different keys with the same
          string representation:

          d = {0: 1, '0': 2}  # JSON key collision
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
        }
        loaded = Dataset._from_loaded_json(
            json.loads(json.dumps(representation)))
        if loaded != representation:
            pprint_file = self.path / f'info.{time()}.pprint'
            pprint_file.write_text(pprint.pformat(representation))
            raise ValueError(f'JSON representation causes data loss, saving '
                             f'into {pprint_file}')
        return representation

    @staticmethod
    def _load_config(conf_path: Path) -> Dict:
        """Get config dictionary from a file. Use this function instead of an
        json.loads, as this function returns correct types for group ids.

        Args:
          conf_path: Path object representing the dataset information (e.g.,
            the return value of Dataset._get_config_path).

        Returns: Dictionary representation of the Dataset.
        """
        return Dataset._from_loaded_json(json.loads(conf_path.read_text()))

    @staticmethod
    def _from_loaded_json(loaded_dict: Dict) -> Dict:
        """Fix types in the datastructure loaded from JSON. Necessary as JSON
        allows only string keys, but for instance group keys are integers in
        Dataset.

        Args:
          loaded_dict: The datastructure returned by json.load on the info.json
            file.

        Returns: The same information with fixed types.
        """
        fixed_dict = copy.deepcopy(loaded_dict)
        find_misspellings(fixed_dict.keys())  # Check for misspellings of keys.
        # Fix type of keys_per_group
        fixed_dict['keys_per_group'] = {
            split: {
                int(group): n_examples
                for group, n_examples in keys_info.items()
            }
            for split, keys_info in loaded_dict['keys_per_group'].items()
        }
        # Fix type of examples_per_group
        fixed_dict['examples_per_group'] = {
            split: {
                int(group): n_examples
                for group, n_examples in ex_info.items()
            }
            for split, ex_info in loaded_dict['examples_per_group'].items()
        }
        # Fix missing keys
        if 'licence' not in fixed_dict:
            # Do not relicence
            fixed_dict['licence'] = ''
        for k in ['firmware_url', 'paper_url']:
            if k not in fixed_dict:
                fixed_dict[k] = ''
        return fixed_dict

    def _write_config(self):
        """Save configuration as json."""
        with open(self._get_config_path(self.path), 'w+') as f:
            json.dump(self._get_config_dictionary(), f)

    @staticmethod
    def from_config(dataset_path: str):
        """Load a dataset from a config file.

        Args:
          dataset_path: The path to the dataset.

        Raises: ValueError if the dataset version is higher than the scaaml
          module used. See scaaml.__version__ docstring.
        """
        dpath = Path(dataset_path)
        conf_path = Dataset._get_config_path(dataset_path)
        cprint(f'reloading {conf_path}', 'magenta')
        config = Dataset._load_config(conf_path)
        # Check that the library version (version of this software) is not
        # lower than what was used to capture the dataset.
        if 'scaaml_version' in config.keys():
            if semver.compare(config['scaaml_version'], scaaml.__version__) > 0:
                raise ValueError(f'SCAAML module is outdated, scaaml_version: '
                                 f'{scaaml.__version__}, but dataset was '
                                 f'created using: {config["scaaml_version"]}')
        return Dataset(
            root_path=str(dpath),
            shortname=config['shortname'],
            architecture=config['architecture'],
            implementation=config['implementation'],
            algorithm=config['algorithm'],
            version=config['version'],
            url=config['url'],
            firmware_url=config['firmware_url'],
            paper_url=config['paper_url'],
            licence=config['licence'],
            description=config['description'],
            firmware_sha256=config['firmware_sha256'],
            measurements_info=config['measurements_info'],
            attack_points_info=config['attack_points_info'],
            capture_info=config['capture_info'],
            compression=config['compression'],
            shards_list=config['shards_list'],
            keys_per_group=config['keys_per_group'],
            keys_per_split=config['keys_per_split'],
            examples_per_group=config['examples_per_group'],
            examples_per_split=config['examples_per_split'],
            examples_per_shard=config['examples_per_shard'],
            min_values=config['min_values'],
            max_values=config['max_values'],
            from_config=True,
        )

    @staticmethod
    def _get_config_path(path) -> Path:
        return Path(path) / 'info.json'

    @staticmethod
    def _cleanup_shards(dataset_path: Path, print_info: bool = True):
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
        for split, slist in config['shards_list'].items():
            kept = 0
            removed = 0
            for shard in slist:
                spath = dataset_path / shard['path']  # The shard.
                if spath.exists():
                    new_shards_list[split].append(shard)
                    kept += 1
                else:
                    removed += 1
            assert kept + removed == len(slist)
            assert len(new_shards_list[split]) == kept
            stats.append([split, kept, removed])

        config['shards_list'] = new_shards_list
        examples_per_split = {}
        examples_per_group = {}
        keys_per_group = {}
        for split, slist in config['shards_list'].items():
            examples_per_split[split] = config['examples_per_shard'] * len(
                config['shards_list'][split])
            # Zero examples per a group is a valid option.
            examples_per_group[split] = {
                k: 0
                for k in config['examples_per_group'][split]
            }
            names_keys_per_group = {
                k: set()
                for k in config['keys_per_group'][split]
            }
            key_names_per_split = set()
            for shard in slist:
                shard_info = Dataset._shard_info_from_name(shard['path'])
                sg = shard_info['shard_group']
                examples_per_group[split][sg] += config['examples_per_shard']
                key_names_per_split.add(shard_info['shard_key'])
                names_keys_per_group[sg].add(shard_info['shard_key'])
            # We suppose that each shard contains at most one key.
            config['keys_per_group'][split] = {
                k: len(names_keys_per_group[k])
                for k in config['keys_per_group'][split]
            }
            config['keys_per_split'][split] = len(key_names_per_split)

        config['examples_per_split'] = examples_per_split
        config['examples_per_group'] = examples_per_group
        if print_info:
            print(tabulate(stats, headers=['split', 'kept', 'removed']))
        return config

    @staticmethod
    def cleanup_shards(dataset_path):
        """Remove non_existing shards from the config and update the config.
        Makes a backup of the old config named info.json.sav.{time()}.json.

        Args:
          dataset_path: The directory of the dataset.

        Example use:
          Dataset.cleanup_shards(
              dataset_path='/mnt/storage/chipwhisperer/test_cleanup')
        """
        fpath = Dataset._get_config_path(dataset_path)

        # Save the old config.
        save_path = Path(f'{str(fpath)}.sav.{time()}.json')
        assert not save_path.exists()  # Do not overwrite.
        save_path.write_text(fpath.read_text())
        cprint("Saving old config to %s" % save_path, 'cyan')

        # Rewrite with the new config.
        cprint("Writing cleaned config", 'green')
        new_config = Dataset._cleanup_shards(dataset_path=Path(dataset_path),
                                             print_info=True)
        with open(fpath, 'w+') as o:
            json.dump(new_config, o)
