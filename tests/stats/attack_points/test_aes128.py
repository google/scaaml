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
"""Test AES128 attack points."""

import pytest

import chipwhisperer.analyzer.attacks.models as cw_models
import numpy as np

from scaaml.aes_forward import AESSBOX
from scaaml.stats.attack_points.aes_128.full_aes import encrypt
from scaaml.stats.attack_points.aes_128.attack_points import *


def test_sub_bytes_in():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)

        computed = np.array([
            SubBytesIn.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            ) for i in range(16)
        ],
                            dtype=np.uint8)

        expected = np.array(
            AESSBOX.sub_bytes_in(key=key, plaintext=plaintext),
            dtype=np.uint8,
        )
        assert (computed == expected).all()


def test_sub_bytes_out():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)

        computed = np.array([
            SubBytesOut.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            ) for i in range(16)
        ],
                            dtype=np.uint8)

        expected = np.array(
            AESSBOX.sub_bytes_out(key=key, plaintext=plaintext),
            dtype=np.uint8,
        )
        assert (computed == expected).all()


def test_last_round_state_diff():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = encrypt(key=key, plaintext=plaintext)

        target_value = [
            LastRoundStateDiff.target_secret(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            ) for i in range(16)
        ]

        for i in range(16):
            computed = LastRoundStateDiff.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            )

            expected = cw_models.aes128_leakage.LastroundStateDiff().leakage(
                pt=plaintext,
                ct=ciphertext,
                key=target_value,
                bnum=i,
            )
            assert computed == expected


@pytest.mark.parametrize("cls", AttackPointAES128.__subclasses__())
def test_target_secret_docstring_promise(cls):
    """Test that the convenience function `target_secret` returns the correct
    guess for all subclasses of AttackPointAES128.
    """
    for _ in range(100):
        byte_index = np.random.randint(0, 16)
        key = np.random.randint(0, 256, 16).astype(np.uint8)
        plaintext = np.random.randint(0, 256, 16).astype(np.uint8)
        ciphertext = AESSBOX.ciphertext(key=key, plaintext=plaintext)

        guess: int = cls.target_secret(
            key=key,
            plaintext=plaintext,
            byte_index=byte_index,
        )
        a: int = cls.leakage_knowing_secrets(
            key=key,
            plaintext=plaintext,
            byte_index=byte_index,
        )
        b: int = cls.leakage_from_guess(
            plaintext=plaintext,
            ciphertext=ciphertext,
            guess=guess,
            byte_index=byte_index,
        )
        assert a == b
