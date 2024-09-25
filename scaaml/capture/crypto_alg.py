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
"""Crypto algorithm."""

from abc import ABC
from typing import Any, Dict, Iterator, Literal, Optional, Type

from scaaml.io import Dataset
from scaaml.aes_forward import AESSBOX

DatasetType = Literal["Training", "Validation"]


class AbstractSCryptoAlgorithm(ABC):
    """Attack points and basic information about it (e.g. key length, etc.)"""

    def __init__(self,
                 *,
                 firmware_sha256: str,
                 crypto_implementation: Type[AESSBOX],
                 purpose: Dataset.SPLIT_T,
                 implementation: str,
                 algorithm: str,
                 examples_per_shard: int,
                 full_kt_filename: str = "parameters_tuples.txt",
                 full_progress_filename: str = "progress_tuples.txt") -> None:
        """Generates a set of key-text pairs and saves those. Does not overwrite
        existing files.

        Args:
          firmware_sha256: SHA256 hash of the binary used on the chip.
          crypto_implementation: The class that provides attack points info and
            attack points values (for instance scaaml.aes_forward.AESSBOX).
          purpose: Type of the dataset. Used in scaaml.io.Dataset.
          implementation: Name of the implementation that was used.
          algorithm: Algorithm name.
          examples_per_shard: Size of a single part (for ML training purposes).
          kt_filename: Filename to save key, text pairs (using resume_kti).
          progress_filename: Filename to save progress (using resume_kti).
          full_kt_filename: The file to save key-text pairs into. This should
            be in the same directory as the whole dataset.
          full_progress_filename: The file to save progress into. This should
            be in the same directory as the whole dataset.
        """
        # When changing the following assert update also
        # SCryptoAlgorithm._dataset
        assert purpose in Dataset.SPLITS
        self._firmware_sha256 = firmware_sha256
        self._crypto_implementation = crypto_implementation
        self._implementation = implementation
        self._algorithm = algorithm

        self._examples_per_shard = examples_per_shard
        self._purpose: Dataset.SPLIT_T = purpose
        self._kt_filename = full_kt_filename
        self._progress_filename = full_progress_filename
        self._full_kt_filename = full_kt_filename
        self._full_progress_filename = full_progress_filename

        # Initialized in a child class.
        self._kti: Optional[Iterator[Any]] = None
        self._stabilization_ktp: Optional[Iterator[Any]] = None

    def attack_points_info(self) -> Dict[str, Dict[str, int]]:
        """Returns the attack points info."""
        return self._crypto_implementation.ATTACK_POINTS_INFO

    def attack_points(self, **kwargs: bytearray) -> Dict[str, bytearray]:
        """Returns the attack points for specific parameters (such as the
        key-text pair).

        Typical usage example:
          aps = crypto_algorithm.attack_points(key=key, plaintext=plaintext)
        """
        c_i = self._crypto_implementation
        aps = {}
        for attack_point_name in c_i.ATTACK_POINTS_INFO:
            aps[attack_point_name] = c_i.get_attack_point(
                attack_point_name, **kwargs)
        return aps

    @property
    def kti(self) -> Optional[Iterator[Any]]:
        """Key-plaintext iterator."""
        return self._kti

    @property
    def stabilization_kti(self) -> Iterator[Any]:
        """Key-text iterator for stabilizing the capture. This is different
        from the real kti.
        """
        assert self._stabilization_ktp is not None
        return self._stabilization_ktp

    @property
    def examples_per_shard(self) -> int:
        """How many traces are captured in a shard."""
        return self._examples_per_shard

    @property
    def key_len(self) -> int:
        """Length of the key in bytes."""
        return self._crypto_implementation.KEY_LENGTH

    @property
    def plaintext_len(self) -> int:
        """Length of the plaintext in bytes."""
        return self._crypto_implementation.PLAINTEXT_LENGTH

    @property
    def firmware_sha256(self) -> str:
        """SHA256 hash of the firmware."""
        return self._firmware_sha256

    @property
    def implementation(self) -> str:
        """The implementation used."""
        return self._implementation

    @property
    def algorithm(self) -> str:
        """The algorithm used."""
        return self._algorithm

    @property
    def purpose(self) -> Dataset.SPLIT_T:
        """The parameter split in scaaml.io.Database.new_shard, in
        Dataset.SPLITS."""
        return self._purpose

    @property
    def _dataset(self) -> DatasetType:
        """Return the dataset type used in ktp_scaaml."""
        # purpose is used in scaaml.io.Dataset
        purpose_to_dataset: Dict[str, DatasetType] = {
            Dataset.TRAIN_SPLIT: "Training",
            Dataset.TEST_SPLIT: "Training",
            Dataset.HOLDOUT_SPLIT: "Validation",
        }
        return purpose_to_dataset[self._purpose]
