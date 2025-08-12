# Copyright 2025 Google LLC
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
"""CPA https://wiki.newae.com/Correlation_Power_Analysis
"""

import math
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tabulate import tabulate

from scaaml.stats.attack_points.aes_128 import LeakageModelAES128


class R:
    """Holds and updates intermediate values.
    """

    def __init__(self, return_absolute_value: bool) -> None:
        """Initialize the computation.

        Args:

          return_absolute_value (bool): If set to True then negative
          correlation is also detected. If set to False only positive
          correlation is detected.
        """
        self.d: int = 0
        self.return_absolute_value: bool = return_absolute_value

        # The following variables are initialized lazily when we know the
        # dimensions of the traces.
        self.sum_h_t: npt.NDArray[np.float64]
        self.sum_h: npt.NDArray[np.float64]
        self.sum_t: npt.NDArray[np.float64]
        self.sum_hh: npt.NDArray[np.float64]
        self.sum_tt: npt.NDArray[np.float64]

    def update(self, trace: npt.NDArray[np.float64],
               hypothesis: npt.NDArray[np.int32]) -> None:
        """Update with another trace.

        Args:

          trace (npt.NDArray[np.float64]): The trace wave observed.

          hypothesis (list[int]): Hypothetical leakage for each possible secret
          value.
        """
        trace = np.array(trace, dtype=np.float64)
        hypothesis = np.array(hypothesis)
        assert len(trace.shape) == 1
        assert len(hypothesis.shape) == 1

        if self.d == 0:
            # Lazy initialize.
            trace_len: int = len(trace)
            hypothesis_possibilities: int = hypothesis.shape[0]
            self.sum_h_t = np.zeros((hypothesis_possibilities, trace_len),
                                    dtype=np.float64)
            self.sum_h = np.zeros(hypothesis_possibilities, dtype=np.float64)
            self.sum_t = np.zeros(trace_len, dtype=np.float64)
            self.sum_hh = np.zeros(hypothesis_possibilities, dtype=np.float64)
            self.sum_tt = np.zeros(trace_len, dtype=np.float64)

        self.d += 1
        self.sum_h_t += np.einsum("i,j->ij", hypothesis, trace)
        self.sum_h += hypothesis
        self.sum_t += trace
        self.sum_hh += hypothesis**2
        self.sum_tt += trace**2

    def guess(self) -> npt.NDArray[np.float64]:
        """Return how much each possible guess value corresponds to the
        observed values. The expected shape is (different_target_secrets,
        trace_len).
        """
        # nominator
        nom = (self.d * self.sum_h_t) - np.einsum("i,j->ij", self.sum_h,
                                                  self.sum_t)

        # denominator squared
        den_a = (self.sum_h**2) - (self.d * self.sum_hh)  # i
        den_b = (self.sum_t**2) - (self.d * self.sum_tt)  # j

        r = nom / np.sqrt(np.einsum("i,j->ij", den_a, den_b))

        if self.return_absolute_value:
            return np.array(np.abs(r), dtype=np.float64)
        else:
            return np.array(r, dtype=np.float64)


class CPA:
    """Do correlation power analysis.
    http://wiki.newae.com/Correlation_Power_Analysis

    This implementation is not optimized for production usage. It might be a
    good idea to use one of the well established implementations.
    """

    def __init__(
        self,
        get_model: Callable[[int], LeakageModelAES128],
        return_absolute_value: bool = True,
        subsample: int = 1,
    ) -> None:
        """Initialize the CPA computation.

        Args:

          get_model (Callable[[int], LeakageModelAES128]): A function for
          turning an index into a leakage model.

          return_absolute_value (bool): If set to True then negative
          correlation is also detected. If set to False only positive
          correlation is detected. The cost is larger ranks (up to twice).
          Defaults to True.

          subsample (int): Update `self.result` only each `subsample` updates
          to save RAM. Defaults to 1 (remember everything).

        Example use:
        ```python
        import numpy as np

        from scaaml.stats.cpa import CPA
        from scaaml.stats.attack_points.aes_128.full_aes import encrypt
        from scaaml.stats.attack_points.aes_128.attack_points import *

        cpa = CPA(get_model=lambda i: LeakageModelAES128(
            byte_index=i,
            attack_point=SubBytesIn(),
            use_hamming_weight=True,
        ))

        key = np.random.randint(0, 256, size=16, dtype=np.uint8)

        # Make sure that both positive and negative correlation works.
        random_signs = np.random.choice(2, 16) * 2 - 1

        for _ in range(100):
            plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)

            # Simulate a trace
            bit_counts = [int(x).bit_count() for x in key ^ plaintext]
            trace = bit_counts + np.random.normal(scale=1.5, size=16)
            trace *= random_signs

            cpa.update(
                trace=trace,
                plaintext=plaintext,
                ciphertext=encrypt(plaintext=plaintext, key=key),
                real_key=key,  # Just to check that the key is constant
            )

        cpa.print_predictions(
            real_key=key,
            plaintext=plaintext,
        )

        cpa.plot_cpa(
            real_key=key,
            plaintext=plaintext,
            experiment_name="cpa_graphs.png",
        )
        ```
        """
        self.models: list[LeakageModelAES128] = [
            get_model(byte_index) for byte_index in range(16)
        ]
        self.result: dict[int, list[list[float]]] = {
            i: [[] for _ in range(256)] for i in range(16)
        }
        self.real_key: Optional[npt.NDArray[np.uint8]] = None
        self.r: list[R] = [
            R(return_absolute_value=return_absolute_value) for _ in range(16)
        ]

        # Sample each `self.subsample` updates.
        if subsample < 1:
            raise ValueError("subsample must be a positive integer.")
        self.subsample: int = subsample
        self.update_counter: int = 0

    def update(self,
               trace: npt.NDArray[np.float32],
               plaintext: npt.NDArray[np.uint8],
               ciphertext: npt.NDArray[np.uint8],
               real_key: Optional[npt.NDArray[np.uint8]] = None) -> None:
        """Update with a new example.

        Args:

          trace (npt.NDArray[np.float32]): The physical measurements (e.g.,
          power, EM over time).

          plaintext (npt.NDArray[np.uint8]): The 16 bytes of input.

          ciphertext (npt.NDArray[np.uint8]): The 16 bytes of output.

          real_key (npt.NDArray[np.uint8]): The secret key. If provided it
          serves to check that all examples have the same secret key.
        """
        if real_key is not None:
            if self.real_key is None:
                self.real_key = real_key
            assert all(self.real_key == real_key)

        for byte in range(16):
            hypothesis: list[int] = [
                self.models[byte].leakage_from_guess(
                    plaintext=plaintext,
                    ciphertext=ciphertext,
                    guess=i,
                ) for i in range(self.models[byte].different_target_secrets)
            ]
            self.r[byte].update(
                trace=trace.astype(np.float64),
                hypothesis=np.array(hypothesis, dtype=np.int32),
            )

            res = self.r[byte].guess()

            # Forget time.
            res = np.max(res, axis=1)
            assert res.shape == (self.models[byte].different_target_secrets,)

            # Fill in the result
            if self.update_counter % self.subsample == 0:
                for value in range(self.models[byte].different_target_secrets):
                    self.result[byte][value].append(float(res[value]))

        self.update_counter += 1

    def print_predictions(self, real_key: npt.NDArray[np.uint8],
                          plaintext: npt.NDArray[np.uint8]) -> None:
        """Print a short prediction summary.

        Args:

          real_key (npt.NDArray[np.uint8]): The real secret key to compare
          against.

          plaintext (npt.NDArray[np.uint8]): The input of AES.
        """
        statistics: dict[str, list[int]] = {
            "byte": [],
            "real": [],
            "guessed": [],
            "rank": [],
        }
        for byte in range(16):
            target_value = self.models[byte].target_secret(
                key=real_key,
                plaintext=plaintext,
            )
            statistics["byte"].append(byte)
            statistics["real"].append(target_value)
            res = np.max(self.r[byte].guess(), axis=1)
            assert res.shape == (self.models[byte].different_target_secrets,)
            statistics["guessed"].append(int(np.argmax(res)))
            # Compute rank
            statistics["rank"].append(int(np.sum(res >= res[target_value])))

        # Print intermediate result
        print()
        current_ranks = statistics["rank"]
        # Estimate of log2 of how many keys we need to try to get the correct
        # one.
        security = math.log2(math.prod(current_ranks))
        print(
            f"Traces: {self.update_counter} mean_rank {np.mean(current_ranks)} "
            f"{security = }")

        print(tabulate([name] + values for name, values in statistics.items()))

    def plot_cpa(self,
                 real_key: npt.NDArray[np.uint8],
                 plaintext: npt.NDArray[np.uint8],
                 experiment_name: str = "cpa.png",
                 logscale: bool = True) -> None:
        """Plot how does the real secret value change position among
        predictions when adding more examples.

        Args:

          real_key (npt.NDArray[np.uint8]): The real secret key to compare
          against.

          plaintext (npt.NDArray[np.uint8]): The input of AES.

          experiment_name (str): The name of the figure being saved. Defaults
          to "cpa.png".

          logscale (bool): Use logarithmic scale on the y axis (the rank axis).
        """
        target_values = [
            self.models[byte].target_secret(key=real_key, plaintext=plaintext)
            for byte in range(16)
        ]
        plt.clf()
        f, arr = plt.subplots(4, 4)
        f.set_size_inches(16, 16, forward=True)
        f.set_dpi(100)
        for byte_i in range(16):
            if logscale:
                arr[byte_i // 4, byte_i % 4].set_yscale("log")

            for value in range(256):
                # skip the correct value
                if value == target_values[byte_i]:
                    continue
                arr[byte_i // 4, byte_i % 4].plot(self.result[byte_i][value],
                                                  "gray")
            arr[byte_i // 4,
                byte_i % 4].plot(self.result[byte_i][target_values[byte_i]],
                                 "red")
            arr[byte_i // 4,
                byte_i % 4].set_xlabel(f"Traces combined for byte_{byte_i:02d}")
        plt.savefig(experiment_name)
        f.clear()
        plt.close(f)
        plt.clf()
