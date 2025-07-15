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
from scaaml.stats.attack_points.aes_128.full_aes import encrypt
from scaaml.stats.attack_points.aes_128.attack_points import *


def cpa_try(figure_path, return_absolute_value, random_correlation_sign):
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
        random_signs = np.random.choice(2, 16) * 2 - 1
    else:
        random_signs = np.ones(shape=16)

    for _ in range(100):
        plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)

        # Simulate a trace
        bit_counts = [int(x).bit_count() for x in key ^ plaintext]
        trace = bit_counts + np.random.normal(scale=1.5, size=16)
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
    for byte in range(16):
        target_value = cpa.models[byte].target_secret(
            key=key,
            plaintext=plaintext,
        )
        res = np.max(cpa.r[byte].guess(), axis=1)
        assert int(np.sum(res >= res[target_value])) <= 2

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
