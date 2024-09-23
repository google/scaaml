# Copyright 2020 Google LLC
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
"""Dataset creation and loading."""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.utils import to_categorical
from termcolor import cprint
from tqdm.auto import tqdm
from glob import glob


# pylint: disable=too-many-positional-arguments
def create_dataset(file_pattern: str,
                   batch_size: int = 32,
                   attack_point: str = "key",
                   attack_byte: int = 0,
                   num_shards: int = 256,
                   num_traces_per_shard: int = 256,
                   max_trace_length: int = 20000,
                   is_training: bool = True,
                   shuffle_size: int = 65535) -> Tuple[Tensor, Tensor]:
    del shuffle_size  # unused
    del is_training  # unused
    del batch_size  # unused

    shards = list_shards(file_pattern, num_shards)
    attack_byte = int(attack_byte)

    if attack_point not in ["key", "sub_bytes_in", "sub_bytes_out"]:
        raise ValueError(
            "invalid attack point. avail: key, sub_bytes_in, sub_bytes_out")

    x_list: List[Tensor] = []
    y_list: List[Tensor] = []
    pb = tqdm(total=num_shards, desc="loading shards")
    with tf.device("/cpu:0"):
        for idx, shard_fname in enumerate(shards):
            x_shard, y_shard = load_shard(shard_fname, attack_byte,
                                          attack_point, max_trace_length,
                                          num_traces_per_shard)

            del idx  # unused
            # if not idx:
            #     x = x_shard
            #     y = y_shard
            # else:
            #     x = tf.concat([x, x_shard], axis=0)
            #     y = tf.concat([y, y_shard], axis=0)
            x_list.append(x_shard)
            y_list.append(y_shard)
            pb.update()
        pb.close()
        x: Tensor = tf.concat(x_list, axis=0)
        y: Tensor = tf.concat(y_list, axis=0)

    cprint("[Generator]", "yellow")
    cprint(f"|-attack point:{attack_point}", "blue")
    cprint(f"|-attack byte:{attack_byte}", "green")
    cprint(f"|-num shards:{num_shards}", "blue")
    cprint(f"|-traces per shards:{num_traces_per_shard}", "green")
    cprint(f"|-y:{str(y.shape)}", "blue")
    cprint(f"|-x:{str(x.shape)}", "green")

    # make it a tf dataset
    # cprint("building tf dataset", "magenta")
    # dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # dataset.cache()
    # if is_training:
    #     dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
    # dataset = dataset.batch(batch_size).prefetch(
    #     tf.data.experimental.AUTOTUNE
    # )
    return (x, y)


def list_shards(file_pattern: str, num_shards: int) -> List[str]:
    return glob(file_pattern)[:num_shards]


# pylint: disable=too-many-positional-arguments
def load_attack_shard(
        fname: str,
        attack_byte: int,
        attack_point: str,
        max_trace_length: int,
        num_traces: int = 256,
        full_key: bool = False) -> Tuple[bytearray, bytearray, Tensor, Tensor]:
    """Load a shard of data that target a given key

    Args:
        fname ([type]): [description]
        attack_byte ([type]): [description]
        attack_point ([type]): [description]
        max_trace_length ([type]): [description]
        num_traces (int, optional): [description]. Defaults to 256.

    Returns:
        list: keys, pts, attack_points_val, power_traces
    """
    del full_key  # unused
    shard = np.load(fname)
    attack_byte = int(attack_byte)

    # key
    k = shard["keys"][attack_byte][:num_traces]
    pts = shard["pts"][attack_byte][:num_traces]
    # load y
    if attack_point == "key":
        y = shard["keys"][attack_byte]
    elif attack_point == "sub_bytes_in":
        y = shard["sub_bytes_in"][attack_byte]
    elif attack_point == "sub_bytes_out":
        y = shard["sub_bytes_out"][attack_byte]
    else:
        raise ValueError(f"Unknown attack point {attack_point}.")

    y = y[:num_traces]
    y = to_categorical(y, 256)
    y = tf.convert_to_tensor(y, dtype="uint8")

    # load x
    x = shard["traces"][:num_traces, :max_trace_length, :]
    x = tf.convert_to_tensor(x, dtype="float32")
    return k, pts, x, y


def load_shard(fname: str, attack_byte: int, attack_point: str,
               max_trace_length: int,
               num_traces_per_shard: int) -> Tuple[Tensor, Tensor]:
    shard = np.load(fname)

    # load y
    if attack_point == "key":
        y = shard["keys"][attack_byte]
    elif attack_point == "sub_bytes_in":
        y = shard["sub_bytes_in"][attack_byte]
    elif attack_point == "sub_bytes_out":
        y = shard["sub_bytes_out"][attack_byte]
    else:
        raise ValueError(f"Unknown attack point {attack_point}.")

    y = y[:num_traces_per_shard]
    y = to_categorical(y, 256)
    y = tf.convert_to_tensor(y, dtype="uint8")

    # load x
    x = shard["traces"][:num_traces_per_shard, :max_trace_length, :]
    x = tf.convert_to_tensor(x, dtype="float32")
    return x, y
