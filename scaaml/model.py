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
"""Model."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, cast

from tabulate import tabulate
from keras.models import Model, load_model
from keras.backend import clear_session

from scaaml.utils import get_model_stub


def get_models_by_attack_point(
        config: Dict[str, Any]) -> DefaultDict[str, List[Optional[Path]]]:
    """Get available models grouped by attack points

    Args:
        config (dict): target config

    Returns:
        dict(list): models grouped by attacks point

    """

    models: DefaultDict[str, List[Optional[Path]]] = defaultdict(list)
    status: Dict[str, str] = {}
    attack_point: str
    for attack_point in config["attack_points"]:
        status[attack_point] = "complete"
        for attack_byte in range(16):
            stub = get_model_stub(attack_point, attack_byte, config)
            model_path = Path("models") / stub
            if not model_path.exists():
                status[str(attack_byte)] = "incomplete"
                models[attack_point].append(None)
                continue
            else:
                models[attack_point].append(model_path)

    rows = [[k, v, len(models[k])] for k, v in status.items()]
    print(
        tabulate(rows,
                 headers=["Attack point", "status", "Num available models"]))
    return models


def load_model_from_idx(models_list: Dict[int, Dict[str, str]],
                        idx: int,
                        verbose: bool = False) -> Any:
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
    path = Path(models_list[idx]["path"])
    return load_model_from_disk(path, verbose=verbose)


def load_model_from_disk(path: Path, verbose: bool = False) -> Any:
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
    assert mdl
    mdl = cast(Model, mdl)  # type: ignore[no-any-unimported]

    # display summary if requested
    if verbose:
        mdl.summary()
    return mdl


@dataclass
class ModelInfo:
    path: Path
    attack_point: str
    attack_byte: int


def get_models_list(config: Dict[str, Any],
                    verbose: bool = False) -> List[ModelInfo]:
    "Return the list of trained models"
    available_models: List[ModelInfo] = []
    for attack_point in config["attack_points"]:
        for attack_byte in config["attack_bytes"]:
            stub = get_model_stub(attack_point, attack_byte, config)
            model_path = Path("models") / stub
            if not model_path.exists():
                continue
            else:
                available_models.append(
                    ModelInfo(path=model_path,
                              attack_point=attack_point,
                              attack_byte=attack_byte))
    if verbose:
        rows = []
        for idx, mdl in enumerate(available_models):
            rows.append([idx, mdl.attack_point, mdl.attack_byte])
        print(tabulate(rows, headers=["model idx", "attack point", "byte"]))
    return available_models
