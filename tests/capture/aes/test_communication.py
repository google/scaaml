from unittest.mock import MagicMock, patch

import chipwhisperer as cw

from scaaml.capture.aes.communication import SCommunication


@patch.object(cw, 'target')
def test_with(mock_target_fn):
    mock_scope = MagicMock()
    mock_target = MagicMock()
    mock_target_fn.return_value = mock_target
    with SCommunication(mock_scope) as target:
        assert target.target.protver == '1.1'
        assert mock_target.dis.call_count == 0

    mock_target.dis.assert_called_once()
