# Copyright 2024 Google LLC
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
"""Compute SNR. Inspired by the tutorial
https://github.com/newaetech/chipwhisperer-jupyter/blob/master/archive/PA_Intro_3-Measuring_SNR_of_Target.ipynb
"""
from collections import defaultdict
from enum import Enum
from typing import Union

import numpy as np

from scaaml.aes_forward import AESSBOX
from scaaml.stats.online import Mean, VarianceSinglePass


class AttackPoint(Enum):
    SUB_BYTES_IN = 0
    SUB_BYTES_OUT = 1


class LeakageModelAES128:
    """The value that might be correlated with value(s) in the trace.
    """

    def __init__(self,
                 byte_index: int = 0,
                 attack_point: AttackPoint = AttackPoint.SUB_BYTES_IN,
                 use_hamming_weight: bool = True) -> None:
        """Gives the leakage function.

        Args:

          byte_index (int): Which byte to target (in range(16)).

          attack_point (AttackPoint): Use either input or output of the first
          SBOX.

          use_hamming_weight (bool): Use just the Hamming weight of the value.
        """
        assert byte_index in range(16)
        self._byte_index: int = byte_index
        self._use_hamming_weight: bool = use_hamming_weight
        self._attack_point: AttackPoint = attack_point

    @staticmethod
    def _safe_cast(
            value: np.typing.NDArray[np.uint8]) -> np.typing.NDArray[np.uint8]:
        """Ideally this function does nothing. But it is very easy to pass an
        array of larger dtype and then be surprised by results of leakage (the
        additional bytes are zero).

        Args:

          value (np.typing.NDArray[np.uint8]): Try to convert this to uint8.

        Raises ValueError if the values cannot be safely converted to uint8.
        """
        uint8_value = np.array(value, dtype=np.uint8)
        if not (uint8_value == value).all():
            raise ValueError("Conversion to uint8 was not successful.")
        return uint8_value

    def leakage(self, plaintext: np.typing.NDArray[np.uint8],
                key: np.typing.NDArray[np.uint8]) -> int:
        """Return the leakage value.

        Args:

          plaintext (np.typing.NDArray[np.uint8]): Array of byte values. The
          method fails if there is a value which cannot be converted to uint8.

          key (np.typing.NDArray[np.uint8]): Array of byte values. The method
          fails if there is a value which cannot be converted to uint8.

        Returns: An integer representing the leakage.
        """
        plaintext = self._safe_cast(plaintext)
        key = self._safe_cast(key)

        # Get the byte value of the leakage.
        byte_value: int
        if self._attack_point == AttackPoint.SUB_BYTES_OUT:
            byte_value = AESSBOX.sub_bytes_out(
                key=bytearray(key),
                plaintext=bytearray(plaintext),
            )[self._byte_index]
        elif self._attack_point == AttackPoint.SUB_BYTES_IN:
            byte_value = AESSBOX.sub_bytes_in(
                key=bytearray(key),
                plaintext=bytearray(plaintext),
            )[self._byte_index]
        else:
            raise NotImplementedError("Unknown attack point "
                                      f"{self._attack_point}")

        # Maybe convert to Hamming weight.
        if self._use_hamming_weight:
            return int(byte_value).bit_count()

        return byte_value


class SNRSinglePass:
    """Single pass SNR implementation.
    """

    def __init__(self,
                 leakage_model: LeakageModelAES128,
                 db: bool = True) -> None:
        """Initialize the SNR.

        Args:

          leakage_model (LeakageModelAES128): What do we expect to be leaking.

          db (bool): If True return decibel (20 * np.log(result)).
        """
        self._leakage_model: LeakageModelAES128 = leakage_model
        self._value_to_variance: defaultdict[
            int, VarianceSinglePass] = defaultdict(VarianceSinglePass)
        self._value_to_mean: defaultdict[int, Mean] = defaultdict(Mean)
        self.db: bool = db

    def update(
        self, example: dict[str, Union[np.typing.NDArray[np.uint8],
                                       np.typing.NDArray[np.float64]]]
    ) -> None:
        """Update itself with another example.

        Args:

          example (dict[str, Union[np.typing.NDArray[np.uint8],
          np.typing.NDArray[np.float64]]]): Assumes that there are "trace1",
          "key", and "plaintext".
        """
        leakage = self._leakage_model.leakage(
            plaintext=example["plaintext"],  # type: ignore[arg-type]
            key=example["key"],  # type: ignore[arg-type]
        )
        trace: np.typing.NDArray[np.float64]
        trace = example["trace1"]  # type: ignore[assignment]
        self._value_to_variance[leakage].update(trace)
        self._value_to_mean[leakage].update(trace)

    @property
    def result(self) -> np.typing.NDArray[np.float64]:
        """Return the SNR values (in time).
        """
        signal_var = np.var(
            np.array([m.result for m in self._value_to_mean.values()]),
            axis=0,
        )

        most_common: int = max(
            ((leak_val, var.n_seen)
             for leak_val, var in self._value_to_variance.items()
             if var.n_seen > 2),
            key=lambda p: p[1])[0]
        noise_var_one_point = self._value_to_variance[most_common].result
        assert noise_var_one_point is not None

        result = signal_var / noise_var_one_point

        if self.db:
            return np.array(20 * np.log(result))

        return np.array(result)
