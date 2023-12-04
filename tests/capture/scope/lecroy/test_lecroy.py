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
"""Test Lecroy."""

from unittest.mock import patch

import scaaml
from scaaml.capture.scope import LeCroy


@patch.object(scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationSocket, "close")
@patch.object(scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationSocket, "query")
@patch.object(scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationSocket, "write")
@patch.object(scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationSocket, "connect")
def test_get_identity_info(mock_connect, mock_write, mock_query, mock_close):
    model = "WAVEMASTER"
    serial_number = "WM01234"
    firmware_level = "1.2.3"
    mock_query.return_value = ",".join([
        "LECROY",
        model,
        serial_number,
        firmware_level,
    ])

    with LeCroy(samples=10,
                offset=0,
                ip_address="127.0.0.1",
                trace_channel="C1",
                trigger_channel="C2",
                communication_timeout=1.0,
                trigger_timeout=0.1) as lecroy:

        assert lecroy.scope.get_identity_info() == {
            "lecroy_model": model,
            "lecroy_serial_number": serial_number,
            "lecroy_firmware_level": firmware_level,
        }
