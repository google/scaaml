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
"""Model."""

from collections import defaultdict
from pathlib import Path
from typing import Dict

from tabulate import tabulate
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from scaaml.utils import get_model_stub


def get_models_by_attack_point(config):
    """Get available models grouped by attack points

    Args:
        config (dict): target config

    Returns:
        dict(list): models grouped by attacks point

    """

    models: defaultdict = defaultdict(list)
    status: Dict = {}
    for attack_point in config["attack_points"]:
        status[attack_point] = "complete"
        for attack_byte in range(16):
            stub = get_model_stub(attack_point, attack_byte, config)
            model_path = f"models/{stub}"
            if not Path(model_path):
                status[attack_byte] = "incomplete"
                models[attack_point].append(None)
                continue
            else:
                models[attack_point].append(model_path)

    rows = [[k, v, len(models[k])] for k, v in status.items()]
    print(
        tabulate(rows,
                 headers=["Attack point", "status", "Num available models"]))
    return models


def load_model_from_idx(models_list, idx, verbose=0):
    """Load a model based of its index id.

    model list is generated via get_model_list()

    Args:
        models_list (list): list of available models
        idx (int): model index
        verbose (int, optional): Display model summary if set to 1.
        Defaults to 0.

    Returns:
        tf.keras.Model
    """
    path = models_list[idx]["path"]
    return load_model_from_disk(path, verbose=verbose)


def load_model_from_disk(path, verbose=0):
    """Load a model based of its index id.

    model list is generated via get_model_list()

    Args:
        path (str): model path
        verbose (int, optional): Display model summary if set to 1.
        Defaults to 0.

    Returns:
        tf.keras.Model: the requested model
    """

    # clear tf graph
    clear_session()

    # load model
    mdl = load_model(path)

    # display summary if requested
    if verbose:
        mdl.summary()
    return mdl


def get_models_list(config, verbose=0):
    "Return the list of trained models"
    available_models = []
    for attack_point in config["attack_points"]:
        for attack_byte in config["attack_bytes"]:
            stub = get_model_stub(attack_point, attack_byte, config)
            model_path = f"models/{stub}"
            if not Path(model_path):
                continue
            else:
                available_models.append({
                    "path": model_path,
                    "attack_point": attack_point,
                    "attack_byte": attack_byte
                })
    if verbose:
        rows = []
        for idx, mdl in enumerate(available_models):
            rows.append([idx, mdl["attack_point"], mdl["attack_byte"]])
        print(tabulate(rows, headers=["model idx", "attack point", "byte"]))
    return available_models
