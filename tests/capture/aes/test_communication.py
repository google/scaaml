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

import chipwhisperer as cw
from chipwhisperer.capture.targets.SimpleSerial import SimpleSerial

from scaaml.capture.aes.communication import CWCommunication


@patch.object(cw, 'target')
def test_with(mock_target_fn):
    mock_scope = MagicMock()
    mock_target_fn.return_value = SimpleSerial()
    with CWCommunication(mock_scope) as target:
        assert isinstance(target.target, SimpleSerial)
        assert target.target.protver == '1.1'
