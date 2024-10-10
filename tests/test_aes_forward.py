# Copyright 2022 Google LLC
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
"""Test AESSBOX."""

import tensorflow as tf

from scaaml.aes import ap_preds_to_key_preds
from scaaml.aes_forward import AESSBOX


def test_bs_in():
    """Test that sub_bytes_in outputs XOR of its inputs."""
    for i in range(256):
        for j in range(256):
            b_i = bytearray([i])
            b_j = bytearray([j])
            assert i ^ j == AESSBOX.sub_bytes_in(b_i, b_j)[0]


def test_bs_out():
    """Test getting output of the SBOX."""
    keys = [
        bytearray([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b
        ]),
        bytearray([
            0xff, 0xfe, 0xfd, 0x00, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b
        ]),
    ]
    texts = [
        bytearray([
            0x03, 0x01, 0x04, 0x01, 0x05, 0x09, 0x02, 0x06, 0x03, 0x01, 0x07,
            0x00
        ]),
        bytearray([
            0x01, 0x03, 0x05, 0x01, 0x07, 0x09, 0x01, 0x00, 0x03, 0x09, 0x0a,
            0x0b
        ]),
    ]
    results = [
        bytearray([
            0x7b, 0x63, 0x6f, 0x77, 0x7c, 0xfe, 0xf2, 0x7c, 0x2b, 0x30, 0xd7,
            0x2b
        ]),
        bytearray([
            0xbb, 0x54, 0x41, 0x7c, 0x7b, 0xfe, 0xc5, 0xc5, 0x2b, 0x63, 0x63,
            0x63
        ]),
    ]

    for key, text, result in zip(keys, texts, results):
        assert result == AESSBOX.sub_bytes_out(key, text)


def test_inverse():
    """Test that ap_preds_to_key_preds . AESSBOX.sub_bytes_out is identity."""
    # Test if ap_preds_to_key_preds is an inverse to AESSBOX.sub_bytes_out.
    keys = [
        bytearray([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b
        ]),
        bytearray([
            0xff, 0xfe, 0xfd, 0x00, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b
        ]),
        bytearray([
            0xff, 0xfe, 0xfd, 0x00, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b
        ]),
        bytearray([
            0xa8, 0x30, 0x11, 0xbf, 0x94, 0x65, 0x31, 0x12, 0x4c, 0x98, 0xdd,
            0xee
        ]),
    ]
    texts = [
        bytearray([
            0x03, 0x01, 0x04, 0x01, 0x05, 0x09, 0x02, 0x06, 0x03, 0x01, 0x07,
            0x00
        ]),
        bytearray([
            0x01, 0x03, 0x05, 0x01, 0x07, 0x09, 0x01, 0x00, 0x03, 0x09, 0x0a,
            0x0b
        ]),
        bytearray([
            0x03, 0x01, 0x04, 0x01, 0x05, 0x09, 0x02, 0x06, 0x03, 0x01, 0x07,
            0x00
        ]),
        bytearray([
            0x93, 0x21, 0x00, 0x41, 0x14, 0x57, 0x32, 0x01, 0xef, 0xcd, 0xab,
            0xef
        ]),
    ]
    for key, text in zip(keys, texts):
        sb_cat = tf.keras.utils.to_categorical(
            AESSBOX.sub_bytes_out(key, text),
            num_classes=256,
        )
        k_cat = tf.keras.utils.to_categorical(
            key,
            num_classes=256,
        )
        assert (ap_preds_to_key_preds(sb_cat, text,
                                      "sub_bytes_out") == k_cat).all()


def test_ciphertext():
    assert AESSBOX.get_attack_point(
        "ciphertext",
        key=bytearray(range(16)),
        plaintext=bytearray(range(16)),
    ) == bytearray(b"\n\x94\x0b\xb5An\xf0E\xf1\xc3\x94X\xc6S\xeaZ")


def test_attack_points():
    """Test getting different attack points using AESSBOX.get_attack_point."""
    key = bytearray([
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b
    ])
    text = bytearray([
        0x03, 0x01, 0x04, 0x01, 0x05, 0x09, 0x02, 0x06, 0x03, 0x01, 0x07, 0x00
    ])
    sub_bytes_out = bytearray([
        0x7b, 0x63, 0x6f, 0x77, 0x7c, 0xfe, 0xf2, 0x7c, 0x2b, 0x30, 0xd7, 0x2b
    ])

    assert AESSBOX.get_attack_point("key", key=key, plaintext=text) == key
    assert AESSBOX.get_attack_point("sub_bytes_in", key=key,
                                    plaintext=text) == AESSBOX.sub_bytes_in(
                                        key=key, plaintext=text)
    assert AESSBOX.get_attack_point("sub_bytes_out", key=key,
                                    plaintext=text) == AESSBOX.sub_bytes_out(
                                        key=key, plaintext=text)
    assert AESSBOX.get_attack_point("sub_bytes_out", key=key,
                                    plaintext=text) == sub_bytes_out
    assert AESSBOX.get_attack_point("plaintext", key=key,
                                    plaintext=text) == text
