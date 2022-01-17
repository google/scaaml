from unittest.mock import MagicMock, patch

import chipwhisperer as cw

from scaaml.capture.aes.scope import SScope


@patch.object(cw, 'scope')
def test_with(mock_scope):
    mock_cwscope = MagicMock()
    mock_scope.return_value = mock_cwscope
    gain = 45
    samples = 7000
    offset = 0
    clock = 7372800
    sample_rate = 'clkgen_x4'
    mock_cwscope.adc.oa.hwInfo.maxSamples.return_value = -1

    with SScope(gain=gain,
                samples=samples,
                offset=offset,
                clock=clock,
                sample_rate=sample_rate) as scope:
        assert scope.scope == mock_cwscope
        assert mock_cwscope.dis.call_count == 0
        assert mock_cwscope.gain.db == gain
        assert mock_cwscope.adc.samples == samples
        assert mock_cwscope.adc.offset == offset
        assert mock_cwscope.adc.basic_mode == "rising_edge"
        assert mock_cwscope.clock.clkgen_freq == clock
        assert mock_cwscope.trigger.triggers == "tio4"
        assert mock_cwscope.clock.adc_src == sample_rate
        assert mock_cwscope.clock.freq_ctr_src == "clkgen"
        assert mock_cwscope.adc.presamples == 0

    assert mock_cwscope.dis.call_count >= 1
