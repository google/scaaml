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
"""Capture script for easier manipulation."""
from pathlib import Path
from typing import Any, Dict, List

from scaaml.aes_forward import AESSBOX
from scaaml.io import Dataset
from scaaml.capture.aes.capture_runner import CaptureRunner
from scaaml.capture.aes.crypto_alg import SCryptoAlgorithm
from scaaml.capture.aes.communication import CWCommunication
from scaaml.capture.aes.control import CWControl
from scaaml.capture.scope import PicoScope, CWScope, DefaultCWScope


def capture_aes_dataset(
        scope_class,
        firmware_sha256: str,
        architecture: str,
        implementation: str,
        shortname: str,
        description: str,
        capture_info: Dict,
        chip_id: int,
        crypto_implementation=AESSBOX,
        algorithm: str = "simpleserial-aes",
        version: int = 1,
        root_path: str = "/mnt/storage/chipwhisperer",
        url: str = "",
        firmware_url: str = "",
        paper_url: str = "",
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        examples_per_shard: int = 64,
        measurements_info=None,
        repetitions: int = 1,
        train_keys: int = 3 * 1024,
        train_plaintexts: int = 256,
        holdout_keys: int = 0 * 1024,
        holdout_plaintexts: int = 1) -> None:
    """Capture or continue capturing the dataset.

    Args:
      scope_class (AbstractSScope): The class of scope that does measurements.
      firmware_sha256: Hash of the used firmware IntelHEX file (HEX files are
        reproducible, ELF binaries are not).
      architecture: Architecture of the used chip.
      implementation: Name of the implementation that was used.
      shortname: Short name of the dataset being captured.
      description: Short description for the scaaml.io.Dataset.
      capture_info: Used as parameters of the scope. Should contain:
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
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
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
      repetitions: Number of captures with a concrete (key, plaintext) pair.
      train_keys: Number of different keys that are used in the train split (set
        to zero not to capture this split), multiple of 256. Also captures
        test split 12.5% size of train (1/8 keys).
      train_plaintexts: Number of different plaintexts used with each key in
        the train split.
      holdout_keys: Number of different keys that are used in the holdout split
        (set to zero not to capture this split), multiple of 256.
      holdout_plaintexts: Number of different plaintexts used with each key in
        the holdout split.

    Raises:
      ValueError: If something is wrong with the splits (either we are
        capturing train and holdout with the same chip, or we are capturing
        an unsupported split).
    """
    # If we are capturing train also capture the test split.
    test_keys = train_keys // 8  # 1/8 = 0.125
    test_plaintexts = train_plaintexts

    if measurements_info is None:
        measurements_info = {
            "trace1": {
                "type": "power",
                "len": capture_info["samples"],
            },
        }

    # Make sure we do not capture train and holdout in the same go (using the
    # same chip).
    if holdout_keys and train_keys:
        raise ValueError("Holdout should not be captured with the same chip as"
                         "test or train")

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
        attack_points_info=crypto_implementation.ATTACK_POINTS_INFO,
        capture_info=capture_info)

    # Check that if we are capturing train or holdout then the other split has
    # different chip_id.
    if holdout_keys:  # Capturing holdout, check train and test.
        if dataset.shards_list[Dataset.TRAIN_SPLIT]:
            if dataset.shards_list[
                    Dataset.TRAIN_SPLIT][0]["chip_id"] == chip_id:
                raise ValueError("Capturing holdout using the same chip as "
                                 "train.")
    if train_keys:  # Capturing train and test, check holdout.
        if dataset.shards_list[Dataset.HOLDOUT_SPLIT]:
            if dataset.shards_list[
                    Dataset.HOLDOUT_SPLIT][0]["chip_id"] == chip_id:
                raise ValueError("Capturing train using the same chip as "
                                 "holdout.")

    # Generators of key-plaintext pairs for different splits.
    crypto_algorithms = []

    def add_crypto_alg(split: Dataset.SPLIT_T, keys: int, plaintexts: int,
                       repetitions: int):
        """Does not overwrite, safe to call multiple times.

        Args:
          split: Which split are we capturing.
          keys: Number of different keys in this split.
          plaintexts: Number of different plaintext captured with each key.
          repetitions: Number of captures of each (key, plaintext) pair.
        """
        if split not in Dataset.SPLITS:
            raise ValueError(
                f"split must be one of {Dataset.SPLITS}, got {split}")
        new_crypto_alg = SCryptoAlgorithm(
            crypto_implementation=crypto_implementation,
            purpose=split,
            implementation=implementation,
            algorithm=algorithm,
            keys=keys,
            plaintexts=plaintexts,
            repetitions=repetitions,
            examples_per_shard=examples_per_shard,
            firmware_sha256=firmware_sha256,
            full_kt_filename=Path(root_path) / dataset.slug /
            f"{split}_parameters_tuples.txt",
            full_progress_filename=Path(root_path) / dataset.slug /
            f"{split}_progress_tuples.txt")
        crypto_algorithms.append(new_crypto_alg)

    if test_keys:
        add_crypto_alg(split=Dataset.TEST_SPLIT,
                       keys=test_keys,
                       plaintexts=test_plaintexts,
                       repetitions=repetitions)
    if train_keys:
        add_crypto_alg(split=Dataset.TRAIN_SPLIT,
                       keys=train_keys,
                       plaintexts=train_plaintexts,
                       repetitions=repetitions)
    if holdout_keys:
        add_crypto_alg(split=Dataset.HOLDOUT_SPLIT,
                       keys=holdout_keys,
                       plaintexts=holdout_plaintexts,
                       repetitions=repetitions)

    if not crypto_algorithms:
        raise ValueError(
            "At least one of [train_keys, holdout_keys] should be non-zero in"
            "order to capture at least one split.")

    # Create context managers and capture dataset.
    _capture(
        scope_class=scope_class,
        capture_info=capture_info,
        chip_id=chip_id,
        crypto_algorithms=crypto_algorithms,
        dataset=dataset,
    )


def _capture(scope_class, capture_info: Dict[str, Any], chip_id: int,
             crypto_algorithms: List, dataset) -> None:
    """Create scope contexts managers and capture the dataset.

    Args:
      scope_class: The class of scope that does measurements.
      capture_info (Dict): Capture information.
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
      crypto_algorithms (List[SCryptoAlgorithm]): List of key, plaintext
        generators.
      dataset (scaaml.io.Dataset): The dataset to save examples to.
    """
    # Capture using PicoScope.
    if scope_class == PicoScope:
        with PicoScope(**capture_info) as picoscope:
            assert picoscope.scope is not None
            with DefaultCWScope() as default_cwscope:
                _control_communication_and_capture(
                    chip_id=chip_id,
                    cwscope=default_cwscope,
                    crypto_algorithms=crypto_algorithms,
                    scope=picoscope,
                    dataset=dataset,
                )
        return  # Everything is finished.

    # Capture using the built-in scope of ChipWhisperer.
    if scope_class == CWScope:
        with CWScope(**capture_info) as scope:
            assert scope.scope is not None
            _control_communication_and_capture(
                chip_id=chip_id,
                cwscope=scope,
                crypto_algorithms=crypto_algorithms,
                scope=scope,
                dataset=dataset,
            )
        return  # Everything is finished.

    # Warn on unknown scope_class.
    raise ValueError(f"Unsupported scope_class: {scope_class}")


def _control_communication_and_capture(chip_id: int, cwscope, crypto_algorithms,
                                       scope, dataset) -> None:
    """Create control and communication context managers and run the capture.

    Args:
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
      cwscope (CWScope): The scope to control.
      crypto_algorithms (List[SCryptoAlgorithm]): List of key, plaintext
        generators.
      scope: The scope that does the measurements.
      dataset (scaaml.io.Dataset): The dataset to save examples to.
    """
    with CWControl(chip_id=chip_id, scope_io=cwscope.scope.io) as control:
        with CWCommunication(cwscope.scope) as target:
            capture_runner = CaptureRunner(crypto_algorithms=crypto_algorithms,
                                           scope=scope,
                                           communication=target,
                                           control=control,
                                           dataset=dataset)
            capture_runner.capture()
