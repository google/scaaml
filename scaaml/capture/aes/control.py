# Copyright 2021-2024 Google LLC
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
"""Control of the chip."""

from time import sleep

from scaaml.capture.control import AbstractSControl

from chipwhisperer.capture.scopes.cwhardware import ChipWhispererExtra


class CWControl(AbstractSControl):
    """turning on/off the chip, resetting the chip, etc."""

    def __init__(self, chip_id: int,
                 scope_io: ChipWhispererExtra.GPIOSettings) -> None:
        """Initialize the control object.

        Args:
          chip_id: Identifies the physical chip/board used. It is unique for a
            single piece of hardware. To identify datasets captured using a
            defective hardware.
          scope_io: The chipwhisperer scope.io
        """
        super().__init__(chip_id=chip_id)
        self._scope_io = scope_io
        self._tio1 = "serial_rx"
        self._tio2 = "serial_tx"
        self._hs2 = "clkgen"

    def reset(self,
              sleep_before: float = 1.0,
              sleep_between: float = 0.5,
              sleep_after: float = 0.5) -> None:
        """Reset the chip.

        Args:
          sleep_before: Time to sleep before turning nrst low (in seconds).
          sleep_between: Time to sleep after turning nrst low (in seconds).
          sleep_after: Time to sleep after turning nrst high_z (in seconds).
        """
        sleep(sleep_before)
        self._scope_io.nrst = "low"
        sleep(sleep_between)
        self._scope_io.nrst = "high_z"
        sleep(sleep_after)

    def initialize(self) -> None:
        """Initialize the target."""
        self._scope_io.tio1 = self._tio1
        self._scope_io.tio2 = self._tio2
        self._scope_io.hs2 = self._hs2

    def turn_on(self) -> None:
        """Turn on the chip."""
        self._scope_io.target_pwr = True

    def turn_off(self) -> None:
        """Turn off the chip."""
        self._scope_io.target_pwr = False
