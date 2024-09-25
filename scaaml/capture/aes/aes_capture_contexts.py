# Copyright 2022-2024 Google LLC
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
"""Capture script for easier manipulation."""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Type

from scaaml.aes_forward import AESSBOX
from scaaml.io import Dataset
from scaaml.capture.aes.capture_runner import CaptureRunner
from scaaml.capture.aes.crypto_alg import SCryptoAlgorithm
from scaaml.capture.aes.communication import CWCommunication
from scaaml.capture.aes.control import CWControl
from scaaml.capture.crypto_alg import AbstractSCryptoAlgorithm
from scaaml.capture.scope import DefaultCWScope
from scaaml.capture.scope.scope_base import AbstractSScope


def capture_aes_dataset(
        *,
        scope_class: Type[AbstractSScope],
        firmware_sha256: str,
        architecture: str,
        implementation: str,
        shortname: str,
        description: str,
        capture_info: Dict[str, Any],
        crypto_implementation: Type[AESSBOX] = AESSBOX,
        algorithm: str = "simpleserial-aes",
        version: int = 1,
        root_path: str = "/mnt/storage/chipwhisperer",
        url: str = "",
        firmware_url: str = "",
        paper_url: str = "",
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        examples_per_shard: int = 64,
        train_iterator: Optional[dict[str, Any]] = None,
        test_iterator: Optional[dict[str, Any]] = None,
        holdout_iterator: Optional[dict[str, Any]] = None,
        measurements_info: Optional[Dict[str, Any]] = None) -> Path:
    """Capture or continue capturing the dataset.

    Args:
      scope_class (AbstractSScope): The class of scope that does measurements.
      firmware_sha256: Hash of the used firmware IntelHEX file (HEX files are
        reproducible, ELF binaries are not).
      architecture: Architecture of the used chip.
      implementation: Name of the implementation that was used.
      shortname: Short name of the dataset being captured.
      description: Short description for the scaaml.io.Dataset.
      capture_info (Dict): Used as parameters of the scope. Should contain:
        samples (int): Number of data points in a single capture = length of the
          trace.
        offset (int): How many samples are discarded between the trigger event
          and the start of the trace.
        sample_rate: Sample rate of the scope. When scope_class is PicoScope
          this is float. When scope_class is CWScope this is string and is the
          clock source for cw.ClockSettings.adc_src (see cw documentation).
        gain (int): Only when scope_class is CWScope. Gain of the scope used
          while capture (see cw documentation).
        clock (int): Only when scope_class is CWScope. CLKGEN output frequency
          (in Hz, see cw documentation).
        holdout_chip_id (int): ID of the chip used for holdout split capture.
        train_chip_id (int): ID of the chip used for train split capture.
        {train_,holdout_}something: Value dependent on the split being
          captured. For capture the prefix is stripped name is also present
          without the prefix. The dictionary cannot have `something` also
          present without the prefix.
      crypto_implementation: The class that provides attack points info and
        attack points values (for instance scaaml.aes_forward.AESSBOX).
      algorithm: Algorithm name.
      version: Version of this dataset.
      root_path: Folder in which the dataset will be stored.
      url: Where to download this dataset.
      firmware_url: Where to download the firmware used for this capture.
      paper_url: Where to find the published paper.
      licence: URL or the whole licence the dataset is published under.
      examples_per_shard: Size of a single part (for ML training purposes).
      measurements_info: Measurements info for scaaml.io.Dataset (what is
        measured, how many points are taken).
      train_iterator (dict[str, Any] | None): If not None passed to
        build_attack_points_iterator.
      test_iterator (dict[str, Any] | None): If not None passed to
        build_attack_points_iterator.
      holdout_iterator (dict[str, Any] | None): If not None passed to
        build_attack_points_iterator.

    Raises:
      ValueError: If something is wrong with the splits (either we are
        capturing train and holdout with the same chip, or we are capturing
        an unsupported split).

    Returns: dataset path of the created dataset.
    """
    if not any([train_iterator, holdout_iterator, test_iterator]):
        raise ValueError(
            "At least one of [train_iterator, holdout_iterator, test_iterator]"
            "should be non-empty in order to capture at least one split.")

    if capture_info["train_chip_id"] == capture_info["holdout_chip_id"]:
        raise ValueError("Cannot have the same chip_id for train and holdout.")

    if measurements_info is None:
        measurements_info = {
            "trace1": {
                "type": "power",
                "len": capture_info["samples"],
            },
        }

    # Create the dataset.
    attack_points_info = crypto_implementation.ATTACK_POINTS_INFO
    dataset = Dataset.get_dataset(
        root_path=root_path,
        shortname=shortname,
        architecture=architecture,
        implementation=implementation,
        algorithm=algorithm,
        version=version,
        firmware_sha256=firmware_sha256,
        description=description,
        url=url,
        firmware_url=firmware_url,
        paper_url=paper_url,
        licence=licence,
        examples_per_shard=examples_per_shard,
        measurements_info=measurements_info,
        attack_points_info=attack_points_info,
        capture_info=capture_info,
    )

    # Capture splits.
    for split, iterator_definition, chip_id in (
        (
            Dataset.TEST_SPLIT,
            test_iterator,
            "train_chip_id",  # same chip as train
        ),
        (
            Dataset.TRAIN_SPLIT,
            train_iterator,
            "train_chip_id",
        ),
        (
            Dataset.HOLDOUT_SPLIT,
            holdout_iterator,
            "holdout_chip_id",
        ),
    ):
        if not iterator_definition:
            # Split should not be captured.
            continue

        crypto_algorithm: SCryptoAlgorithm = SCryptoAlgorithm(
            iterator_definition=iterator_definition,
            crypto_implementation=crypto_implementation,
            purpose=split,
            implementation=implementation,
            algorithm=algorithm,
            examples_per_shard=examples_per_shard,
            firmware_sha256=firmware_sha256,
            full_kt_filename=str(
                Path(root_path) / dataset.slug /
                f"{split}_parameters_tuples.txt"),
            full_progress_filename=str(
                Path(root_path) / dataset.slug /
                f"{split}_progress_tuples.txt"),
        )

        prefix, sep, _ = chip_id.partition("_")
        assert prefix in ("train", "holdout")
        current_capture_info = _get_current_capture_info(
            capture_info,
            prefix=prefix + sep,  # type: ignore[arg-type]
        )

        _capture(
            scope_class=scope_class,
            capture_info=current_capture_info,
            chip_id=capture_info[chip_id],
            crypto_algorithms=[crypto_algorithm],
            dataset=dataset,
        )

    return dataset.path


def _get_current_capture_info(
        capture_info: Dict[str, Any],
        prefix: Literal["train_", "holdout_"]) -> Dict[str, Any]:
    """Update capture info for use of capturing train or holdout. Fill the
    following based on the current prefix value (train_ or holdout_): chip_id,
    cw_scope_serial_number, trace_channel, trigger_pin.

    Args:
      capture_info (Dict): The old capture info.
      prefix (str): Prefix that is going to be stripped.
    """
    current_capture_info = deepcopy(capture_info)
    for name, value in capture_info.items():
        if name.startswith(prefix):
            short_name = name.removeprefix(prefix)

            # Check that we do not set short_name by mistake
            if short_name in current_capture_info:
                msg = f"Cannot have both {name} and {short_name} in " \
                      f"capture_info. The convention is to have only {name} " \
                      f"with the {prefix} when it depends on the split " \
                      f"being captured."
                raise ValueError(msg)

            # Provide also the short value
            current_capture_info[short_name] = value
    return current_capture_info


def _capture(scope_class: Type[AbstractSScope], capture_info: Dict[str, Any],
             chip_id: int,
             crypto_algorithms: Sequence[AbstractSCryptoAlgorithm],
             dataset: Any) -> None:
    """Create scope contexts managers and capture the dataset.

    Args:
      scope_class: The class of scope that does measurements.
      capture_info (Dict): Capture information.
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
      crypto_algorithms (Sequence[SCryptoAlgorithm]): Sequence of key,
        plaintext generators.
      dataset (scaaml.io.Dataset): The dataset to save examples to.
    """
    with scope_class(**capture_info) as scope_context:
        # Save information acquired from the scope
        scope_context.post_init(dataset=dataset)

        assert scope_context.scope is not None
        with DefaultCWScope(
                capture_info.get("cw_scope_serial_number")) as default_cwscope:
            _control_communication_and_capture(
                chip_id=chip_id,
                cwscope=default_cwscope,
                crypto_algorithms=crypto_algorithms,
                scope=scope_context,
                dataset=dataset,
            )


def _control_communication_and_capture(
        chip_id: int, cwscope: DefaultCWScope,
        crypto_algorithms: Sequence[AbstractSCryptoAlgorithm],
        scope: AbstractSScope, dataset: Any) -> None:
    """Create control and communication context managers and run the capture.

    Args:
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
      cwscope (CWScope): The scope to control.
      crypto_algorithms (Sequence[SCryptoAlgorithm]): Sequence of key,
        plaintext generators.
      scope: The scope that does the measurements.
      dataset (scaaml.io.Dataset): The dataset to save examples to.
    """
    scope_parameter = cwscope.scope
    with CWControl(
            chip_id=chip_id,
            scope_io=scope_parameter.io,  # type: ignore[attr-defined]
    ) as control:
        with CWCommunication(cwscope.scope) as target:
            capture_runner = CaptureRunner(crypto_algorithms=crypto_algorithms,
                                           scope=scope,
                                           communication=target,
                                           control=control,
                                           dataset=dataset)
            capture_runner.capture()
