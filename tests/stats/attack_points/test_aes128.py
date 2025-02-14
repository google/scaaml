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

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
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

        cipher = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        ciphertext = bytearray(
            encryptor.update(plaintext) + encryptor.finalize())

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

        # Always targetting a byte.
        assert cls.different_target_secrets() == 256


@pytest.mark.parametrize("ap_cls", AttackPointAES128.__subclasses__())
def test_leakage_model_id(ap_cls):
    leakage_model = LeakageModelAES128(
        byte_index=0,
        attack_point=ap_cls(),
        use_hamming_weight=False,
    )

    assert leakage_model.different_target_secrets == 256
    assert leakage_model.different_leakage_values == 256


@pytest.mark.parametrize("ap_cls", AttackPointAES128.__subclasses__())
def test_leakage_model_hw(ap_cls):
    byte_index: int = 0
    leakage_model = LeakageModelAES128(
        byte_index=byte_index,
        attack_point=ap_cls(),
        use_hamming_weight=True,
    )

    assert leakage_model.different_target_secrets == 256
    assert leakage_model.different_leakage_values == 9

    plaintext = np.arange(1, 17, dtype=np.uint8)
    key = np.arange(16, dtype=np.uint8)

    assert leakage_model.leakage_knowing_secrets(
        plaintext=plaintext,
        key=key,
    ) in range(leakage_model.different_leakage_values)
    assert leakage_model.leakage_from_guess(
        plaintext=plaintext,
        ciphertext=encrypt(plaintext=plaintext, key=key),
        guess=0,
    ) in range(leakage_model.different_leakage_values)

    target = leakage_model.target_secret(plaintext=plaintext, key=key)
    assert leakage_model.leakage_knowing_secrets(
        plaintext=plaintext,
        key=key,
    ) == leakage_model.leakage_from_guess(
        plaintext=plaintext,
        ciphertext=encrypt(plaintext=plaintext, key=key),
        guess=target,
    )
