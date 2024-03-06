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
"""Utils common to various SCAAML components"""

from multiprocessing import Pool
from random import randint
from typing import Any, Dict, List, Sequence
import time

from glob import glob
from termcolor import cprint
from termcolor._types import Color
from tqdm.auto import tqdm
from chipwhisperer.common.traces import Trace
import numpy as np
import numpy.typing as npt
import tensorflow as tf


def pretty_hex(val: int) -> str:
    "convert a value into a pretty hex"
    s = hex(int(val))
    s = s[2:]  # remove 0x
    if len(s) == 1:
        s = "0" + s
    return s.upper()


def bytelist_to_hex(lst: Sequence[int], spacer: str = " ") -> str:
    h = []

    for e in lst:
        h.append(pretty_hex(e))
    return spacer.join(h)


def hex_display(lst: Sequence[int],
                prefix: str = "",
                color: Color = "green") -> None:
    "display a list of int as colored hex"
    h = []
    if len(prefix) > 0:
        prefix += "\t"
    for e in lst:
        h.append(pretty_hex(e))
    cprint(prefix + " ".join(h), color)


def get_model_stub(attack_point: str, attack_byte: int,
                   config: Dict[str, str]) -> str:
    device = config["device"]
    algorithm = config["algorithm"]
    model = config["model"]
    version = config["version"]
    max_trace_len = config["max_trace_len"]
    return (f"{device}-{algorithm}-{model}-v{version}-ap_{attack_point}-"
            f"byte_{attack_byte}-len_{max_trace_len}")


def get_target_stub(config: Dict[str, str]) -> str:
    device = config["device"]
    algorithm = config["algorithm"]
    return f"{device}-{algorithm}"


def get_num_gpu() -> int:
    return len(tf.config.list_physical_devices("GPU"))


def tf_cap_memory() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def convert_shard_to_cw(info: Dict[str, Any]) -> List[Trace]:
    # avoid trashing the HD by de-synchronizing multi process
    time.sleep(randint(0, 100) / 1000)
    cw_traces = []
    shard = np.load(info["fname"])
    # CW traces
    cts = np.transpose(shard["cts"])
    pts = np.transpose(shard["pts"])
    keys = np.transpose(shard["keys"])

    for idx in range(info["num_traces_by_shard"]):
        wave = np.squeeze(shard["traces"][idx])
        wave = wave[:info["trace_len"]]

        t = Trace(wave, pts[idx], cts[idx], keys[idx])
        cw_traces.append(t)
    return cw_traces


def convert_to_chipwhisperer_format(file_pattern: str, num_shards: int,
                                    num_traces_by_shard: int,
                                    trace_len: int) -> List[Trace]:

    filenames = glob(file_pattern)[:num_shards]
    num_traces = len(filenames) * num_traces_by_shard

    # creating info for multiprocessing
    chunks = []
    for fname in filenames:
        chunks.append({
            "fname": fname,
            "num_traces_by_shard": num_traces_by_shard,
            "trace_len": trace_len
        })

    with Pool() as p:
        cw_traces = []
        pb = tqdm(total=num_traces, desc="Converting", unit="traces")
        for traces in p.imap_unordered(convert_shard_to_cw, chunks):
            cw_traces.extend(traces)
            pb.update(num_traces_by_shard)

        pb.close()
        return cw_traces


def display_config(config_name: str, config: Dict[str, str]) -> None:
    """Pretty print a config object in terminal.

    Args:
        config_name (str): name of the config
        config (dict): config to display
    """
    cprint(f"[{config_name}]", "magenta")
    cnt = 1
    for k, v in config.items():
        color: Color = "yellow"
        if cnt % 2:
            color = "cyan"
        cprint(f"{k}:{v}", color)
        cnt += 1


def from_categorical(predictions: Sequence[npt.ArrayLike]) -> List[np.intp]:
    "reverse of categorical"
    # note: doing it as a list is significantly faster than a single argmax
    return [np.argmax(p) for p in predictions]
