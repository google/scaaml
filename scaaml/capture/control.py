# Copyright 2022-2024 Google LLC
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

from abc import ABC, abstractmethod
from time import sleep
from types import TracebackType
from typing import Optional
from typing_extensions import Self


class AbstractSControl(ABC):
    """turning on/off the chip, resetting the chip, etc."""

    def __init__(self, chip_id: int):
        """Initialize the control object.

        Args:
          chip_id: Identifies the physical chip/board used. It is unique for a
            single piece of hardware. To identify datasets captured using a
            defective hardware.
        """
        self._chip_id = chip_id

    def __enter__(self) -> Self:
        """Turn on, initialize, and reset the chip."""
        self.turn_on()
        self.initialize()
        self.reset()
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Turn off the chip. Exceptions fall through."""
        self.turn_off()

    @abstractmethod
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

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the target."""

    @abstractmethod
    def turn_on(self) -> None:
        """Turn on the chip."""

    @abstractmethod
    def turn_off(self) -> None:
        """Turn off the chip."""

    def power_cycle(self,
                    sleep_before: float = 0.0,
                    sleep_between: float = 0.5,
                    sleep_after: float = 0.5) -> None:
        """Turn off, sleep, turn on, sleep.

        Args:
          sleep_before: Time to sleep before turning off (in seconds).
          sleep_between: Time to sleep after turning off (in seconds).
          sleep_after: Time to sleep after turning on again (in seconds).
        """
        sleep(sleep_before)
        self.turn_off()
        sleep(sleep_between)
        self.turn_on()
        sleep(sleep_after)

    @property
    def chip_id(self) -> int:
        """Chip id is not expected to change."""
        return self._chip_id
