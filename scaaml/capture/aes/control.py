"""Control of the chip."""

from time import sleep


class SControl:
    """turning on/off the chip, resetting the chip, etc."""
    def __init__(self, chip_id: int, scope_io) -> None:
        """Initialize the control object.

        Args:
          chip_id: Identifies the physical chip/board used. It is unique for a
            single piece of hardware. To identify datasets affected captured
            using a defective hardware.
        """
        self._chip_id = chip_id
        self._scope_io = scope_io
        self._tio1 = "serial_rx"
        self._tio2 = "serial_tx"
        self._hs2 = "clkgen"

    def __enter__(self):
        """Turn on and reset the chip. Initialize target."""
        self.turn_on()
        self._scope_io.tio1 = self._tio1
        self._scope_io.tio2 = self._tio2
        self._scope_io.hs2 = self._hs2
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Turn off the chip. Exceptions fall through."""
        self.turn_off()

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
        self._scope_io.nrst = 'low'
        sleep(sleep_between)
        self._scope_io.nrst = 'high_z'
        sleep(sleep_after)

    def turn_on(self):
        """Turn on the chip."""
        pass

    def turn_off(self):
        """Turn off the chip."""
        self._scope_io.target_pwr = False

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
