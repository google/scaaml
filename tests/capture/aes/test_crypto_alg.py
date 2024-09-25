# Copyright 2021 Google LLC
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

from unittest.mock import MagicMock, patch

from scaaml.aes_forward import AESSBOX
from scaaml.capture.aes.crypto_alg import SCryptoAlgorithm
from scaaml.io import resume_kti


@patch.object(resume_kti, 'create_resume_kti')
@patch.object(resume_kti, 'ResumeKTI')
def test_init(mock_resumekti, mock_create_resume_kti):
    description = 'train'
    implementation = 'MBEDTLS'
    algorithm = 'simpleserial-aes'
    examples_per_shard = 64
    firmware_sha256 = 'TODO'
    full_kt_filename = 'key_text_filename.txt'
    full_progress_filename = 'progress_filename.txt'

    crypto_alg = SCryptoAlgorithm(
        crypto_implementation=AESSBOX,
        iterator_definition={
            "operation":
                "cartesian_product",
            "operands": [{
                "operation":
                    "constants",
                "name":
                    "key",
                "length":
                    16,
                "values": [[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
                ]]
            }, {
                "operation": "balanced_generator",
                "name": "plaintext",
                "length": 16,
                "bunches": 8,
                "elements": 256
            }]
        },
        purpose=description,
        implementation=implementation,
        algorithm=algorithm,
        examples_per_shard=examples_per_shard,
        firmware_sha256=firmware_sha256,
        full_kt_filename=full_kt_filename,
        full_progress_filename=full_progress_filename,
    )

    assert mock_create_resume_kti.call_count == 2
    kwargs = mock_create_resume_kti.call_args.kwargs
    assert kwargs['shard_length'] == examples_per_shard
    assert kwargs['kt_filename'] == full_kt_filename
    assert kwargs['progress_filename'] == full_progress_filename
    assert crypto_alg.examples_per_shard == examples_per_shard
    assert crypto_alg.key_len == 16
    stab_kti = crypto_alg.stabilization_kti
    parameters = next(stab_kti)
    assert crypto_alg.key_len == 16
    assert crypto_alg.plaintext_len == 16
    assert crypto_alg.firmware_sha256 == firmware_sha256
    assert crypto_alg.implementation == implementation
    assert crypto_alg.algorithm == algorithm
    assert crypto_alg.purpose == description


@patch.object(resume_kti, 'create_resume_kti')
@patch.object(resume_kti, 'ResumeKTI')
def test_attack_points(mock_resumekti, mock_create_resume_kti):
    description = 'train'
    implementation = 'MBEDTLS'
    algorithm = 'simpleserial-aes'
    examples_per_shard = 64
    firmware_sha256 = 'TODO'
    full_kt_filename = 'key_text_filename.txt'
    full_progress_filename = 'progress_filename.txt'
    crypto_alg = SCryptoAlgorithm(
        crypto_implementation=AESSBOX,
        iterator_definition={
            "operation":
                "cartesian_product",
            "operands": [{
                "operation":
                    "constants",
                "name":
                    "key",
                "length":
                    16,
                "values": [[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
                ]]
            }, {
                "operation": "balanced_generator",
                "name": "plaintext",
                "length": 16,
                "bunches": 8,
                "elements": 256
            }]
        },
        purpose=description,
        implementation=implementation,
        algorithm=algorithm,
        examples_per_shard=examples_per_shard,
        firmware_sha256=firmware_sha256,
        full_kt_filename=full_kt_filename,
        full_progress_filename=full_progress_filename,
    )
    key = bytearray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    plaintext = bytearray(
        [255, 254, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap = {
        'plaintext': plaintext,
        'sub_bytes_in': AESSBOX.sub_bytes_in(key=key, plaintext=plaintext),
        'sub_bytes_out': AESSBOX.sub_bytes_out(key=key, plaintext=plaintext),
        'ciphertext': AESSBOX.ciphertext(key=key, plaintext=plaintext),
        'key': key,
    }
    assert crypto_alg.attack_points(key=key, plaintext=plaintext) == ap


@patch.object(resume_kti, 'create_resume_kti')
@patch.object(resume_kti, 'ResumeKTI')
def test_attack_points_info(mock_resumekti, mock_create_resume_kti):
    description = 'train'
    implementation = 'MBEDTLS'
    algorithm = 'simpleserial-aes'
    examples_per_shard = 64
    firmware_sha256 = 'TODO'
    full_kt_filename = 'key_text_filename.txt'
    full_progress_filename = 'progress_filename.txt'
    crypto_alg = SCryptoAlgorithm(
        crypto_implementation=AESSBOX,
        iterator_definition={
            "operation":
                "cartesian_product",
            "operands": [{
                "operation":
                    "constants",
                "name":
                    "key",
                "length":
                    16,
                "values": [[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
                ]]
            }, {
                "operation": "balanced_generator",
                "name": "plaintext",
                "length": 16,
                "bunches": 8,
                "elements": 256
            }]
        },
        purpose=description,
        implementation=implementation,
        algorithm=algorithm,
        examples_per_shard=examples_per_shard,
        firmware_sha256=firmware_sha256,
        full_kt_filename=full_kt_filename,
        full_progress_filename=full_progress_filename,
    )
    max_val = 256
    api = {
        'sub_bytes_in': {
            'len': 16,
            'max_val': max_val,
        },
        'sub_bytes_out': {
            'len': 16,
            'max_val': max_val,
        },
        'key': {
            'len': 16,
            'max_val': max_val,
        },
        'plaintext': {
            'len': 16,
            'max_val': max_val,
        },
        'ciphertext': {
            'len': 16,
            'max_val': max_val,
        },
    }
    assert crypto_alg.attack_points_info() == api
