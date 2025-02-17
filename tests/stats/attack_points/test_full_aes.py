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
"""Test full AES128."""

import numpy as np

from scaaml.aes_forward import AESSBOX
from scaaml.capture.input_generators import build_attack_points_iterator
from scaaml.stats.attack_points.aes_128.full_aes import *
from scaaml.stats.attack_points.aes_128.attack_points import LastRoundStateDiff


def test_encrypt():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = AESSBOX.ciphertext(key=key, plaintext=plaintext)
        ciphertext = np.array(ciphertext, dtype=np.uint8)

        assert (encrypt(key=key, plaintext=plaintext) == ciphertext).all()


def test_decrypt():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = AESSBOX.ciphertext(key=key, plaintext=plaintext)
        ciphertext = np.array(ciphertext, dtype=np.uint8)

        assert (decrypt(key=key, ciphertext=ciphertext) == plaintext).all()


def test_encrypt_decrypt():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = encrypt(key=key, plaintext=plaintext)
        assert (decrypt(key=key, ciphertext=ciphertext) == plaintext).all()


def test_encrypt_with_states():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = encrypt(key=key, plaintext=plaintext)

        intermediates = encrypt_with_states(
            key=key,
            plaintext=plaintext,
        )

        # state
        assert intermediates["state"].shape == (41 * 16,)

        # key_schedule
        assert intermediates["key_schedule"].shape == (44 * 4,)

        # key
        assert (intermediates["key"] == key).all()

        # plaintext
        assert (intermediates["plaintext"] == plaintext).all()

        # known states
        assert (intermediates["state"][:16] == plaintext).all()
        assert (intermediates["state"][-16:] == ciphertext).all()
        assert (intermediates["state"][16:32] == AESSBOX.sub_bytes_in(
            key=key,
            plaintext=plaintext,
        )).all()
        assert (intermediates["state"][32:48] == AESSBOX.sub_bytes_out(
            key=key,
            plaintext=plaintext,
        )).all()
        # known last round state diff
        expected = np.array([
            LastRoundStateDiff.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            ) for i in range(16)
        ],
                            dtype=np.uint8)

        r = 40
        st10_permuted = np.array(intermediates["state"][r * 16:(r + 1) * 16],
                                 dtype=np.uint8)
        r = 37
        st9_permuted = np.array(intermediates["state"][r * 16:(r + 1) * 16],
                                dtype=np.uint8)
        # The permutation is optional, but we stay consistent with chipwhisperer
        computed = (st9_permuted ^ st10_permuted)[[
            0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
        ]]

        assert (computed == expected).all()


def test_balanced_last_round_diff_generator():
    rng = np.random.default_rng()
    description = {
        "operation":
            "cartesian_product",
        "operands": [
            {
                "operation":
                    "constants",
                "name":
                    "state_9",
                "length":
                    16,
                "values": [[
                    247, 124, 110, 82, 233, 174, 78, 165, 216, 0, 81, 9, 148,
                    124, 23, 246
                ]]
            },
            {
                "operation": "balanced_generator",
                "name": "state_10",
                "length": 16,
                "bunches": 1,
                "elements": 256,
            },
        ]
    }
    counter = np.zeros(256)
    for inputs in build_attack_points_iterator(description):
        state_9 = np.array(inputs["state_9"], dtype=np.uint8)
        state_10 = np.array(inputs["state_10"], dtype=np.uint8)

        # The following works but there is no control over the key.
        ciphertext = state_10[np.argsort(
            [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])]
        last_round_key = shift_rows(
            sub_bytes(state_9.reshape((4, 4), order="F"))).reshape(
                (16,), order="F") ^ state_10
        key = key_schedule_inv(last_round_key[np.argsort(
            [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6,
             11])])[:4].reshape(-1, order="C")

        plaintext = decrypt(ciphertext=ciphertext, key=key)

        byte_index = 0
        values = np.array([
            LastRoundStateDiff.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=byte_index,
            ) for byte_index in range(16)
        ],
                          dtype=np.uint8)
        expected = state_9 ^ state_10
        assert (values == expected).all()
        counter[values[0]] += 1

    each_byte_repeated: int = len(
        build_attack_points_iterator(description)) / 256
    assert (counter == each_byte_repeated).all()


def test_encrypt_with_states_latin_square():
    rng = np.random.default_rng()
    for _ in range(100):
        plaintext = rng.integers(256, size=16, dtype=np.uint8)
        key = rng.integers(256, size=16, dtype=np.uint8)
        ciphertext = encrypt(key=key, plaintext=plaintext)

        intermediates = encrypt_with_states(
            key=key,
            plaintext=plaintext,
        )

        st10 = ciphertext[[
            0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
        ]]

        last_round_key = key_schedule(key)[-4:].copy().reshape(16, order="C")[[
            0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
        ]]
        st9 = sub_bytes_inv(
            shift_rows_inv((st10 ^ last_round_key).reshape(
                (4, 4), order="F"))).reshape((16,), order="F")

        computed = st9 ^ st10

        # known last round state diff
        expected = np.array([
            LastRoundStateDiff.leakage_knowing_secrets(
                key=key,
                plaintext=plaintext,
                byte_index=i,
            ) for i in range(16)
        ],
                            dtype=np.uint8)

        assert (computed == expected).all()


def test_key_schedule_inv():
    rng = np.random.default_rng()
    for _ in range(100):
        key = rng.integers(256, size=16, dtype=np.uint8)
        schedule = key_schedule(key)
        last_key_round = schedule[-4:].copy().reshape(16, order="C")
        W_inv = key_schedule_inv(last_key_round)

        assert (W_inv[-4:] == schedule[-4:]).all()
        assert (W_inv[-8:] == schedule[-8:]).all()
        assert (schedule == W_inv).all()
