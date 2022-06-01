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

from unittest.mock import MagicMock

from scaaml.capture.aes.control import CWControl


def test_control():
    chip_id = 314159
    with CWControl(chip_id=chip_id, scope_io=MagicMock()) as control:
        assert control.chip_id == chip_id
        assert control._scope_io.tio1 == "serial_rx"
        assert control._scope_io.tio2 == "serial_tx"
        assert control._scope_io.hs2 == "clkgen"
        assert control._scope_io.nrst == "high_z"
