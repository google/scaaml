from unittest.mock import MagicMock

from scaaml.capture.aes.control import SControl


def test_control():
    chip_id = 314159
    with SControl(chip_id=chip_id, scope_io=MagicMock()) as control:
        assert control.chip_id == chip_id
        assert control._scope_io.tio1 == "serial_rx"
        assert control._scope_io.tio2 == "serial_tx"
        assert control._scope_io.hs2 == "clkgen"
        assert control._scope_io.nrst == 'high_z'
