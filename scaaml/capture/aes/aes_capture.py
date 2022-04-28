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

"""Capture script for easier manipulation."""
from pathlib import Path
from typing import Dict, Literal

from scaaml.aes_forward import AESSBOX
from scaaml.io import Dataset
from scaaml.capture.aes.capture_runner import CaptureRunner
from scaaml.capture.aes.crypto_alg import SCryptoAlgorithm
from scaaml.capture.aes.communication import SCommunication
from scaaml.capture.aes.control import SControl
from scaaml.capture.aes.scope import SScope


def capture_aes_dataset(
        firmware_sha256: str,
        architecture: str,
        implementation: str,
        shortname: str,
        description: str,
        capture_info: Dict,
        crypto_implementation=AESSBOX,
        algorithm: str = 'simpleserial-aes',
        version: int = 1,
        root_path: str = '/mnt/storage/chipwhisperer',
        url: str = '',
        firmware_url: str = '',
        paper_url: str = '',
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        examples_per_shard: int = 64,
        measurements_info=None,
        repetitions: int = 1,
        chip_id: int = 164019,
        test_keys: int = 1024,
        test_plaintexts: int = 256,
        train_keys: int = 3 * 1024,
        train_plaintexts: int = 256,
        holdout_keys: int = 0 * 1024,
        holdout_plaintexts: int = 1) -> None:
    """Capture or continue capturing the dataset.

    Args:
      firmware_sha256: Hash of the used firmware binary.
      architecture: Architecture of the used chip.
      implementation: Name of the implementation that was used.
      shortname: Short name of the dataset being captured.
      description: Short description for the scaaml.io.Dataset.
      capture_info: Used as parameters of scaaml.io.SScope. Should contain:
        gain: int: Gain of the scope used while capture (see cw documentation).
        samples: int: Number of data points in a single capture = length of the
          trace (see cw documentation).
        offset: int: After how many samples to start recording (see cw
          documentation).
        clock: int: CLKGEN output frequency (in Hz, see cw documentation).
        sample_rate: str: Clock source for cw.ClockSettings.adc_src (see cw
          documentation).
      crypto_implementation: The class that provides attack points info and
        attack points values (for instance scaaml.aes_forward.AESSBOX).
      algorithm: Algorithm name.
      version: Version of this dataset.
      root_path: Folder in which the dataset will be stored.
      url: Where to download this dataset.
      firmware_url: Where to dowload the firmware used while capture.
      paper_url: Where to find the published paper.
      licence: URL or the whole licence the dataset is published under.
      examples_per_shard: Size of a single part (for ML training purposes).
      measurements_info: Measurements info for scaaml.io.Dataset (what is
        measured, how many points are taken).
      repetitions: Number of captures with a concrete (key, plaintext) pair.
      chip_id: Identifies the physical chip/board used. It is unique for a
        single piece of hardware. To identify datasets affected captured
        using a defective hardware.
      test_keys: Number of different keys that are used in the test split (set
        to zero not to capture this split).
      test_plaintexts: Number of different plaintexts used with each key in the test split.
      train_keys: Number of different keys that are used in the train split (set
        to zero not to capture this split), multiple of 256.
      train_plaintexts: Number of different plaintexts used with each key in the train split.
      holdout_keys: Number of different keys that are used in the holdout split (set
        to zero not to capture this split), multiple of 256.
      holdout_plaintexts: Number of different plaintexts used with each key in the holdout split.
    """

    if measurements_info is None:
        measurements_info = {
            "trace": {
                "type": "power",
                "len": capture_info['samples'],
            },
        }

    if holdout_keys and (test_keys or train_keys):
        raise ValueError('Holdout should not be captured with the same chip as'
                         'test or train')

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

    # Generators of key-plaintext pairs for different splits.
    crypto_algorithms = []

    def add_crypto_alg(split: Literal['test', 'train', 'holdout'], keys: int,
                       plaintexts: int, repetitions: int):
        """Does not overwrite, safe to call multiple times.

        Args:
          split: Which split are we capturing.
          keys: Number of different keys in this split.
          plaintexts: Number of different plaintext captured with each key.
          repetitions: Number of captures of each (key, plaintext) pair.
        """
        allowed_splits = ['test', 'train', 'holdout']
        if split not in allowed_splits:
            raise ValueError(
                f'split must be one of {allowed_splits}, got {split}')
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
            f'{split}_key_text_pairs.txt',
            full_progress_filename=Path(root_path) / dataset.slug /
            f'{split}_progress_pairs.txt')
        crypto_algorithms.append(new_crypto_alg)

    if test_keys:
        add_crypto_alg(split='test',
                       keys=test_keys,
                       plaintexts=test_plaintexts,
                       repetitions=repetitions)
    if train_keys:
        add_crypto_alg(split='train',
                       keys=train_keys,
                       plaintexts=train_plaintexts,
                       repetitions=repetitions)
    if holdout_keys:
        add_crypto_alg(split='holdout',
                       keys=holdout_keys,
                       plaintexts=holdout_plaintexts,
                       repetitions=repetitions)

    if not crypto_algorithms:
        raise ValueError(
            'At least one of [test_keys, train_keys, holdout_keys] should be '
            'non-zero in order to capture at least one split.')

    with SScope(**capture_info) as scope:
        assert scope.scope is not None
        with SControl(chip_id=chip_id, scope_io=scope.scope.io) as control:
            with SCommunication(scope.scope) as target:
                capture_runner = CaptureRunner(
                    crypto_algorithms=crypto_algorithms,
                    scope=scope,
                    communication=target,
                    control=control,
                    dataset=dataset)
                capture_runner.capture()


def capture_aes_scald_stm32f4_mbedtls(
        firmware_sha256: str,
        crypto_implementation=AESSBOX,
        algorithm: str = 'simpleserial-aes',
        version: int = 1,
        root_path: str = '/mnt/storage/chipwhisperer',
        url: str = '',
        firmware_url: str = '',
        paper_url: str = '',
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        examples_per_shard: int = 64,
        measurements_info=None,
        repetitions: int = 1,
        chip_id: int = 164019,
        test_keys: int = 1024,
        test_plaintexts: int = 256,
        train_keys: int = 3 * 1024,
        train_plaintexts: int = 256,
        holdout_keys: int = 0 * 1024,
        holdout_plaintexts: int = 1) -> None:
    """SCALD STM32F4 MBEDTLS specific defaults for capture_aes_dataset."""
    architecture: str = 'CW308_STM32F4'
    implementation: str = 'MBEDTLS'
    shortname: str = 'aes_scald'
    description: str = 'SCALD AES'

    # Parameters for the scope
    gain: int = 45
    samples: int = 7000
    offset: int = 0
    clock: int = 7372800
    sample_rate: str = 'clkgen_x4'
    capture_info = {
        'gain': gain,
        'samples': samples,
        'offset': offset,
        'clock': clock,
        'sample_rate': sample_rate,
    }

    capture_aes_dataset(firmware_sha256=firmware_sha256,
                        architecture=architecture,
                        implementation=implementation,
                        shortname=shortname,
                        description=description,
                        crypto_implementation=crypto_implementation,
                        algorithm=algorithm,
                        version=version,
                        root_path=root_path,
                        url=url,
                        firmware_url=firmware_url,
                        paper_url=paper_url,
                        licence=licence,
                        examples_per_shard=examples_per_shard,
                        measurements_info=measurements_info,
                        capture_info=capture_info,
                        repetitions=repetitions,
                        chip_id=chip_id,
                        test_keys=test_keys,
                        test_plaintexts=test_plaintexts,
                        train_keys=train_keys,
                        train_plaintexts=train_plaintexts,
                        holdout_keys=holdout_keys,
                        holdout_plaintexts=holdout_plaintexts)
