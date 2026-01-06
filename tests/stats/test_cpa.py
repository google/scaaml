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

import numpy as np

import pytest

from scaaml.stats.cpa import CPA
from scaaml.stats.cpa.cpa import CPA as CPA_NP
from scaaml.stats.cpa.cpa_jax import CPA as CPA_JAX
from scaaml.stats.attack_points.aes_128.full_aes import encrypt
from scaaml.stats.attack_points.aes_128.attack_points import *


@pytest.mark.slow
@pytest.mark.parametrize("random_correlation_sign", [True, False])
@pytest.mark.parametrize("return_absolute_value", [True, False])
@pytest.mark.parametrize("use_hamming_weight", [True, False])
@pytest.mark.parametrize("attack_point_cls", AttackPointAES128.all_subclasses())
def test_cpa_with_leakage_model(
    random_correlation_sign,
    return_absolute_value,
    use_hamming_weight,
    attack_point_cls,
    tmp_path,
):
    if attack_point_cls == Plaintext:
        # Plaintext provides no information for us.
        return

    figure_path = tmp_path

    trace_len: int = 17
    cpa = CPA(
        get_model=lambda i: LeakageModelAES128(
            byte_index=i,
            attack_point=attack_point_cls(),
            use_hamming_weight=use_hamming_weight,
        ),
        return_absolute_value=return_absolute_value,
    )

    key = np.random.randint(0, 256, size=16, dtype=np.uint8)

    # Make sure that both positive and negative correlation works (might give
    # 2* worse ranks).
    if random_correlation_sign:
        random_signs = np.random.choice(2, trace_len) * 2 - 1
    else:
        random_signs = np.ones(shape=trace_len)

    for _ in range(100):
        plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)

        # Simulate a trace
        bit_counts = [
            cpa.models[i].leakage_knowing_secrets(plaintext=plaintext, key=key)
            for i in range(16)
        ]
        bit_counts.extend([0] * (trace_len - len(bit_counts)))
        trace = bit_counts + np.random.normal(scale=1.5, size=trace_len)
        # np.bitwise_count requires NumPy>=2, CW requires <2
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

    max_rank: int = 0
    res = cpa.guess_no_time()
    for byte in range(16):
        target_value = cpa.models[byte].target_secret(
            key=key,
            plaintext=plaintext,
        )
        max_rank = max(int(np.sum(res[byte] >= res[byte][target_value])),
                       max_rank)
    if random_correlation_sign and not return_absolute_value:
        assert max_rank > 20
    else:
        assert max_rank <= 2

    cpa.plot_cpa(
        real_key=key,
        plaintext=plaintext,
        experiment_name=str(figure_path),
    )


def cpa_results_close(
    random_correlation_sign,
    return_absolute_value,
    use_hamming_weight,
    attack_point_cls,
):
    if attack_point_cls == Plaintext:
        # Plaintext provides no information for us.
        return

    trace_len: int = 17
    cpa_np = CPA_NP(
        get_model=lambda i: LeakageModelAES128(
            byte_index=i,
            attack_point=attack_point_cls(),
            use_hamming_weight=use_hamming_weight,
        ),
        return_absolute_value=return_absolute_value,
    )
    cpa_jax = CPA_JAX(
        get_model=lambda i: LeakageModelAES128(
            byte_index=i,
            attack_point=attack_point_cls(),
            use_hamming_weight=use_hamming_weight,
        ),
        return_absolute_value=return_absolute_value,
    )

    key = np.random.randint(0, 256, size=16, dtype=np.uint8)

    # Make sure that both positive and negative correlation works (might give
    # 2* worse ranks).
    if random_correlation_sign:
        random_signs = np.random.choice(2, trace_len) * 2 - 1
    else:
        random_signs = np.ones(shape=trace_len)

    for _ in range(100):
        plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)

        # Simulate a trace
        bit_counts = [
            cpa_np.models[i].leakage_knowing_secrets(
                plaintext=plaintext,
                key=key,
            ) for i in range(16)
        ]
        bit_counts.extend([0] * (trace_len - len(bit_counts)))
        trace = bit_counts + np.random.normal(scale=1.5, size=trace_len)
        # np.bitwise_count requires NumPy>=2, CW requires <2
        trace *= random_signs

        cpa_np.update(
            trace=trace,
            plaintext=plaintext,
            ciphertext=encrypt(plaintext=plaintext, key=key),
            real_key=key,  # Just to check that the key is constant
        )
        cpa_jax.update(
            trace=trace,
            plaintext=plaintext,
            ciphertext=encrypt(plaintext=plaintext, key=key),
            real_key=key,  # Just to check that the key is constant
        )

    np.testing.assert_allclose(
        cpa_jax.guess(),
        cpa_np.guess(),
        atol=1e-5,
        rtol=0.2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("random_correlation_sign", [True, False])
@pytest.mark.parametrize("return_absolute_value", [True, False])
@pytest.mark.parametrize("use_hamming_weight", [True, False])
@pytest.mark.parametrize("attack_point_cls", AttackPointAES128.all_subclasses())
def test_cpa_results_close_slow(
    random_correlation_sign,
    return_absolute_value,
    use_hamming_weight,
    attack_point_cls,
):
    cpa_results_close(
        random_correlation_sign=random_correlation_sign,
        return_absolute_value=return_absolute_value,
        use_hamming_weight=use_hamming_weight,
        attack_point_cls=attack_point_cls,
    )


def test_cpa_results_close_fast():
    cpa_results_close(
        random_correlation_sign=False,
        return_absolute_value=False,
        use_hamming_weight=True,
        attack_point_cls=SubBytesIn,
    )


def cpa_try(figure_path, return_absolute_value, random_correlation_sign):
    trace_len: int = 23
    cpa = CPA(
        get_model=lambda i: LeakageModelAES128(
            byte_index=i,
            attack_point=SubBytesIn(),
            use_hamming_weight=True,
        ),
        return_absolute_value=return_absolute_value,
    )

    key = np.random.randint(0, 256, size=16, dtype=np.uint8)

    # Make sure that both positive and negative correlation works (might give
    # 2* worse ranks).
    if random_correlation_sign:
        random_signs = np.random.choice(2, trace_len) * 2 - 1
    else:
        random_signs = np.ones(shape=trace_len)

    for _ in range(100):
        plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)

        # Simulate a trace
        bit_counts = [int(x).bit_count() for x in key ^ plaintext]
        bit_counts.extend([0] * (trace_len - len(bit_counts)))
        trace = bit_counts + np.random.normal(scale=1.5, size=trace_len)
        # np.bitwise_count requires NumPy>=2, CW requires <2
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
    res = cpa.guess_no_time()
    for byte in range(16):
        target_value = cpa.models[byte].target_secret(
            key=key,
            plaintext=plaintext,
        )
        assert int(np.sum(res[byte] >= res[byte][target_value])) <= 2

    cpa.plot_cpa(
        real_key=key,
        plaintext=plaintext,
        experiment_name=str(figure_path),
    )


def test_cpa(tmp_path):
    # This shall pass no matter the correlation sign.
    cpa_try(
        figure_path=tmp_path,
        return_absolute_value=True,
        random_correlation_sign=True,
    )


def test_cpa_positive_correlation(tmp_path):
    # This shall pass since there are both positive and negative correlation.
    cpa_try(
        figure_path=tmp_path,
        return_absolute_value=False,
        random_correlation_sign=False,
    )


def test_cpa_wrong_correlation(tmp_path):
    # This shall fail since some correlation is likely to be negative.
    with pytest.raises(AssertionError):
        cpa_try(
            figure_path=tmp_path,
            return_absolute_value=False,
            random_correlation_sign=True,
        )
