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

import numpy as np
from unittest.mock import MagicMock, patch

import chipwhisperer as cw

from scaaml.capture.aes.capture_runner import CaptureRunner
from scaaml.capture.aes.crypto_input import CryptoInput


@patch.object(cw, 'capture_trace')
def test_capture_trace(mock_capture_trace):
    m_crypto_alg = MagicMock()
    m_scope = MagicMock()
    m_communication = MagicMock()
    m_control = MagicMock()
    m_dataset = MagicMock()
    capture_runner = CaptureRunner(crypto_algorithms=[m_crypto_alg],
                                   scope=m_scope,
                                   communication=m_communication,
                                   control=m_control,
                                   dataset=m_dataset)
    key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                   dtype=np.uint8)
    plaintext = np.array([255, 254, 0, 3, 4, 5, 6, 7, 8, 9, 15, 1, 2, 3, 1, 5],
                         dtype=np.uint8)
    mock_capture_trace.return_value.textin = bytearray(plaintext)
    crypto_input = CryptoInput((key, plaintext))
    capture_runner.capture_trace(crypto_input=crypto_input)
    mock_capture_trace.assert_called_once_with(scope=m_scope.scope,
                                               target=m_communication.target,
                                               plaintext=bytearray(plaintext),
                                               key=bytearray(key))


@patch.object(CaptureRunner, 'capture_trace')
def test_get_attack_points_and_measurement(mock_capture_trace):
    key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plaintext = [255, 254, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    crypto_input = CryptoInput((key, plaintext))
    trace = MagicMock()
    trace.textin = bytearray(plaintext)
    mock_capture_trace.side_effect = [None, False, trace]
    m_crypto_alg = MagicMock()
    m_scope = MagicMock()
    m_communication = MagicMock()
    m_control = MagicMock()
    m_dataset = MagicMock()
    capture_runner = CaptureRunner(crypto_algorithms=[m_crypto_alg],
                                   scope=m_scope,
                                   communication=m_communication,
                                   control=m_control,
                                   dataset=m_dataset)
    ap, measurement = capture_runner.get_attack_points_and_measurement(
        crypto_alg=m_crypto_alg, crypto_input=crypto_input)
    assert mock_capture_trace.call_count == 3
    m_crypto_alg.attack_points.assert_called_once_with(
        plaintext=bytearray(plaintext), key=bytearray(key))
    assert measurement == {
        "trace1": trace.wave,
    }


@patch.object(CaptureRunner, 'get_attack_points_and_measurement')
def test_stabilize_capture(mock_capture_trace):
    mock_capture_trace.return_value = (MagicMock(), MagicMock())
    m_crypto_alg = MagicMock()
    k = MagicMock()
    t = MagicMock()
    m_crypto_alg.get_stabilization_kti.return_value = iter(((k, t),))
    m_scope = MagicMock()
    m_communication = MagicMock()
    m_control = MagicMock()
    m_dataset = MagicMock()
    capture_runner = CaptureRunner(crypto_algorithms=[m_crypto_alg],
                                   scope=m_scope,
                                   communication=m_communication,
                                   control=m_control,
                                   dataset=m_dataset)
    capture_runner._stabilize_capture(crypto_alg=m_crypto_alg)
    m_crypto_alg.get_stabilization_kti.assert_called_once_with()
    assert mock_capture_trace.call_count >= 5


@patch.object(CaptureRunner, '_stabilize_capture')
@patch.object(CaptureRunner, '_capture_dataset')
def test_capture(m_capture, m_stabilize):
    m_crypto_alg = MagicMock()
    m_scope = MagicMock()
    m_communication = MagicMock()
    m_control = MagicMock()
    m_dataset = MagicMock()
    capture_runner = CaptureRunner(crypto_algorithms=[m_crypto_alg],
                                   scope=m_scope,
                                   communication=m_communication,
                                   control=m_control,
                                   dataset=m_dataset)
    capture_runner.capture()
    m_stabilize.assert_called_once_with(crypto_alg=m_crypto_alg)
    m_capture.assert_called_once_with(crypto_alg=m_crypto_alg)
    m_dataset.check.assert_called_once_with()


@patch.object(CaptureRunner, 'get_attack_points_and_measurement')
def test_capture_dataset(mock_gapam):
    m_crypto_alg = MagicMock()
    keys = 3072
    examples_per_shard = 64
    plaintexts = 256
    repetitions = 1
    m_crypto_alg.examples_per_shard = examples_per_shard
    m_crypto_alg.plaintexts = plaintexts
    m_crypto_alg.repetitions = repetitions
    m_crypto_alg.kti = [
        (x % 256, x % 256) for x in range(keys * plaintexts * repetitions)
    ]
    m_scope = MagicMock()
    m_communication = MagicMock()
    m_control = MagicMock()
    m_dataset = MagicMock()
    m_attack_points = MagicMock()
    m_measurement = MagicMock()
    mock_gapam.return_value = (m_attack_points, m_measurement)
    capture_runner = CaptureRunner(crypto_algorithms=[m_crypto_alg],
                                   scope=m_scope,
                                   communication=m_communication,
                                   control=m_control,
                                   dataset=m_dataset)
    capture_runner._capture_dataset(crypto_alg=m_crypto_alg)
    m_dataset.close_shard.assert_called_once_with()
    assert m_dataset.new_shard.call_count == len(
        m_crypto_alg.kti) // examples_per_shard
    assert m_dataset.write_example.call_count == len(m_crypto_alg.kti)
