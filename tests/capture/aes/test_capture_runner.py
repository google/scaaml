from unittest.mock import MagicMock, patch
import chipwhisperer as cw

from scaaml.capture.aes.capture_runner import CaptureRunner


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
    key = bytearray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    plaintext = bytearray(
        [255, 254, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    capture_runner.capture_trace(key=key, plaintext=plaintext)
    mock_capture_trace.assert_called_once_with(scope=m_scope.scope,
                                               target=m_communication.target,
                                               plaintext=plaintext,
                                               key=key)


@patch.object(CaptureRunner, 'capture_trace')
def test_get_attack_points_and_measurement(mock_capture_trace):
    key = bytearray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    plaintext = bytearray(
        [255, 254, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    trace = MagicMock()
    trace.textin = plaintext
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
        key=key, plaintext=plaintext, crypto_alg=m_crypto_alg)
    assert mock_capture_trace.call_count == 3
    m_crypto_alg.attack_points.assert_called_once_with(plaintext=plaintext,
                                                       key=key)
    assert measurement == {
        "trace": trace.wave,
    }


@patch.object(CaptureRunner, 'capture_trace')
def test_stabilize_capture(mock_capture_trace):
    m_crypto_alg = MagicMock()
    k = MagicMock()
    t = MagicMock()
    m_crypto_alg.get_stabilization_kti.return_value = iter(((k, t), ))
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
    m_crypto_alg.kti = [(x % 256, x % 256)
                        for x in range(keys * plaintexts * repetitions)]
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
