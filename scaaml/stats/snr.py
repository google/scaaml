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
from typing import Union

import numpy as np

from scaaml.stats.attack_points.aes_128.attack_points import LeakageModelAES128
from scaaml.stats.online import Mean, VarianceSinglePass


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
        leakage = self._leakage_model.leakage_knowing_secrets(
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
