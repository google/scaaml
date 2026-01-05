# Copyright 2026 Google LLC
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

from typing import Callable

import numpy as np
import numpy.typing as npt

from scaaml.stats.attack_points.aes_128 import LeakageModelAES128
from scaaml.stats.cpa.base_cpa import CPABase


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
        # Number of seen traces D
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

        # D (so far)
        self.d += 1
        # i indexes the hypothesis possible values
        # j indexes the time dimension

        # \sum_{d=1}^{D} h_{d,i} t_{d,j}
        self.sum_h_t += np.einsum("i,j->ij", hypothesis, trace)

        # \sum_{d=1}^{D} h_{d, i}
        self.sum_h += hypothesis

        # \sum_{d=1}^{D} t_{d, j}
        self.sum_t += trace

        # \sum_{d=1}^{D} h_{d, i}^2
        self.sum_hh += hypothesis**2

        # \sum_{d=1}^{D} t_{d, j}^2
        self.sum_tt += trace**2

    def guess(self) -> npt.NDArray[np.float64]:
        """Return how much each possible guess value corresponds to the
        observed values. The expected shape is (different_target_secrets,
        trace_len).
        """
        # http://wiki.newae.com/Correlation_Power_Analysis
        # r_{i, j} = \frac{
        #    D \sum_{d=1}^{D} h_{d,i} t_{d,j}
        #    - \sum_{d=1}^{D} h_{d,i} \sum_{d=1}^{D} t_{d,j}
        # }{
        #    \sqrt{
        #        \left( (\sum_{d=1}^{D} h_{d,i} )^2
        #               - D \sum_{d=1}^{D} h_{d,i}^2 \right)
        #        \cdot
        #        \left( (\sum_{d=1}^{D} t_{d,j} )^2
        #               - D \sum_{d=1}^{D} t_{d,j}^2 \right)
        #    }
        # }

        # nom_{i,j} = D self.sum_h_t
        #             - \sum_{d=1}^{D} h_{d,i} \sum_{d=1}^{D} t_{d,j}
        nom = (self.d * self.sum_h_t) - np.einsum("i,j->ij", self.sum_h,
                                                  self.sum_t)

        # denominator squared
        den_a = (self.sum_h**2) - (self.d * self.sum_hh)  # i
        den_b = (self.sum_t**2) - (self.d * self.sum_tt)  # j

        r = nom / np.sqrt(np.einsum("i,j->ij", den_a, den_b))

        if self.return_absolute_value:
            return np.array(np.abs(r), dtype=np.float64)
        else:
            return r


class CPA(CPABase):
    """Do correlation power analysis using pure NumPy implementation.
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
        super().__init__(
            get_model=get_model,
            return_absolute_value=return_absolute_value,
            subsample=subsample,
        )
        self.r: list[R] = [
            R(return_absolute_value=return_absolute_value) for _ in range(16)
        ]

    def _guess(self) -> npt.NDArray[np.float32]:
        return np.array(
            [self.r[byte_index].guess() for byte_index in range(16)],
            dtype=np.float32,
        )

    def _update(
        self,
        trace: npt.NDArray[np.float32],
        hypothesis: npt.NDArray[np.uint32],
    ) -> None:
        assert len(self.r) == len(hypothesis)
        trace = np.array(trace, dtype=np.float64)

        for byte_index in range(16):
            self.r[byte_index].update(
                trace=trace,
                hypothesis=hypothesis[byte_index],
            )
