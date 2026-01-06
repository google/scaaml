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
"""Accelerated CPA implementation based on
https://google.github.io/sedpack/tutorials/sca/gpu_cpa_template/
and
https://wiki.newae.com/Correlation_Power_Analysis
"""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from scaaml.stats.attack_points.aes_128 import LeakageModelAES128
from scaaml.stats.cpa.base_cpa import CPABase


class UpdateData(NamedTuple):
    """A pytree representing the current update.

    Attributes:

      trace (ArrayLike): The trace for this example, shape (trace_len,).

      hypothesis (ArrayLike): The leakage value given the guess. Assumed to be
      in range(different_leakage_values). The shape is
      (different_target_secrets,).
    """
    trace: ArrayLike
    hypothesis: ArrayLike


def get_initial_aggregate_multi_byte(
    trace_len: int,
    different_target_secrets: int = 256,
    num_byte_indexes: int = 16,
) -> dict[str, ArrayLike]:
    """Return an initial aggregate for a all byte indices at once.

    Args:

      trace_len (int): The length of a single trace (or number of points of
      interest if you cut the trace).

      different_target_secrets (int): How many values can the secret have. Most
      likely we are trying to infer a byte value (even when the leakage model
      is Hamming weight).

      num_byte_indexes (int): Defaults to 16 but could be more, e.g., in case
      of AES256.

    Returns: A pytree representing state of online CPA computation for all
    byte indices.

    Keys and values:

      d (ArrayLike): The number of seen examples, shape (1,).

      sum_h_t (ArrayLike): The running sum outer products of hypothesis values
      and trace, shape (num_byte_indexes, different_target_secrets, trace_len).

      sum_h (ArrayLike): The running sum of all hypothesis, shape
      (num_byte_indexes, different_target_secrets,).

      sum_hh (ArrayLike): The running sum of squares of all hypothesis values,
      shape (num_byte_indexes, different_target_secrets,).

      sum_t (ArrayLike): The running sum of all traces, shape (trace_len,).

      sum_tt (ArrayLike): The running sum of squares all traces, shape
      (trace_len,).
    """
    dtype = jnp.float32
    return {
        "d":
            jnp.zeros(1, dtype=jnp.int64),
        "sum_h_t":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets, trace_len),
                dtype=dtype,
            ),
        "sum_h":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets),
                dtype=dtype,
            ),
        "sum_hh":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets),
                dtype=dtype,
            ),
        "sum_t":
            jnp.zeros(
                trace_len,
                dtype=dtype,
            ),
        "sum_tt":
            jnp.zeros(
                trace_len,
                dtype=dtype,
            ),
    }


@jax.jit
def r_update(
    state: dict[str, ArrayLike],
    data: UpdateData,
) -> Tuple[dict[str, ArrayLike], jnp.int32]:
    """Update the CPA aggregate state.
    """
    # Check the dimensions if debugging. This will work even across vmap, jit,
    # scan, etc.
    assert data.trace.shape == state["sum_t"].shape  # type: ignore[union-attr]
    assert data.hypothesis.shape == state[
        "sum_h"].shape  # type: ignore[union-attr]

    # D (so far)
    d = state["d"] + 1
    # i indexes the hypothesis possible values
    # j indexes the time dimension

    # \sum_{d=1}^{D} h_{d,i} t_{d,j}
    sum_h_t = state["sum_h_t"] + jnp.einsum("i,j->ij", data.hypothesis,
                                            data.trace)

    # \sum_{d=1}^{D} h_{d, i}
    sum_h = state["sum_h"] + data.hypothesis

    # \sum_{d=1}^{D} t_{d, j}
    sum_t = state["sum_t"] + data.trace

    # \sum_{d=1}^{D} h_{d, i}^2
    sum_hh = state["sum_hh"] + data.hypothesis**2

    # \sum_{d=1}^{D} t_{d, j}^2
    sum_tt = state["sum_tt"] + data.trace**2

    return (
        {
            "d": d,
            "sum_h_t": sum_h_t,
            "sum_h": sum_h,
            "sum_hh": sum_hh,
            "sum_t": sum_t,
            "sum_tt": sum_tt,
        },
        d,
    )


@partial(jax.jit, static_argnames=["return_absolute_value"])
def r_guess_with_time(
    state: dict[str, ArrayLike],
    return_absolute_value: bool,
) -> ArrayLike:
    """Free standing version of `CPA.guess`.
    """
    num_byte_indexes = 16
    different_target_secrets = 256
    trace_len = state["sum_h_t"].shape[-1]
    assert state["d"].shape == (1,)  # type: ignore[union-attr]
    assert state["sum_h_t"].shape == (  # type: ignore[union-attr]
        num_byte_indexes,
        different_target_secrets,
        trace_len,
    )
    assert state["sum_h"].shape == (  # type: ignore[union-attr]
        num_byte_indexes,
        different_target_secrets,
    )  # type: ignore[union-attr]
    assert state["sum_hh"].shape == (  # type: ignore[union-attr]
        num_byte_indexes,
        different_target_secrets,
    )  # type: ignore[union-attr]
    assert state["sum_t"].shape == (trace_len,)  # type: ignore[union-attr]
    assert state["sum_tt"].shape == (trace_len,)  # type: ignore[union-attr]

    nom = (state["d"] * state["sum_h_t"]) - jnp.einsum(
        "ij,k->ijk", state["sum_h"], state["sum_t"])

    # denominator squared
    den_a = (state["sum_h"]**2) - (state["d"] * state["sum_hh"])  # i
    den_b = (state["sum_t"]**2) - (state["d"] * state["sum_tt"])  # j

    r = nom / jnp.sqrt(jnp.einsum("ij,k->ijk", den_a, den_b))

    if return_absolute_value:
        return jnp.abs(r)
    return r


@partial(jax.jit, static_argnames=["return_absolute_value"])
def r_guess_no_time(
    state: dict[str, ArrayLike],
    return_absolute_value: bool,
) -> ArrayLike:
    # Forget time axis.
    return jnp.max(
        r_guess_with_time(
            state,
            return_absolute_value=return_absolute_value,
        ),
        axis=-1,
    )


class CPA(CPABase):
    """Do correlation power analysis using JAX.
    http://wiki.newae.com/Correlation_Power_Analysis

    See https://google.github.io/sedpack/tutorials/sca/gpu_cpa_template/ for
    implementation details and
    https://google.github.io/sedpack/tutorials/sca/gpu_cpa_template/#technical-considerations
    for notes on running several attacks in parallel.

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
        super().__init__(
            get_model=get_model,
            return_absolute_value=return_absolute_value,
            subsample=subsample,
        )

        # AES128
        self._num_byte_indexes: int = 16

        self.aggregate: dict[str, ArrayLike] | None = None

        self.aggregate_vmap = {
            "d": None,
            "sum_h_t": 0,
            "sum_h": 0,
            "sum_hh": 0,
            "sum_t": None,
            "sum_tt": None,
        }
        self.r_update_multi_index = jax.jit(
            jax.vmap(
                r_update,
                in_axes=(
                    self.aggregate_vmap,
                    UpdateData(
                        trace=None,  # type: ignore[arg-type]
                        hypothesis=0,
                    ),
                ),
                out_axes=(
                    self.aggregate_vmap,
                    None,
                ),
            ))

    def _update(
        self,
        trace: npt.NDArray[np.float32],
        hypothesis: npt.NDArray[np.int32],
    ) -> None:
        if self.aggregate is None:
            self.aggregate = get_initial_aggregate_multi_byte(
                trace_len=len(trace),
                different_target_secrets=256,  # Predicting a byte value.
                num_byte_indexes=self._num_byte_indexes,
            )

        self.aggregate, _ = self.r_update_multi_index(
            self.aggregate,
            UpdateData(
                trace=jnp.array(trace, dtype=jnp.float32),
                hypothesis=jnp.array(hypothesis, dtype=jnp.int32),
            ),
        )

    def guess(self) -> npt.NDArray[np.float32]:
        assert self.aggregate is not None
        return np.array(r_guess_with_time(
            state=self.aggregate,
            return_absolute_value=self.return_absolute_value,
        ),
                        dtype=np.float32)
