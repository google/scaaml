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

from scaaml.capture.scope import CWScope


@patch.object(cw, "scope")
def test_with(mock_scope):
    mock_cwscope = MagicMock()
    mock_scope.return_value = mock_cwscope
    gain = 45
    samples = 7000
    offset = 0
    clock = 7372800
    sample_rate = "clkgen_x4"
    mock_cwscope.adc.oa.hwInfo.maxSamples.return_value = -1
    capture_info = {
        "gain": gain,
        "samples": samples,
        "offset": offset,
        "clock": clock,
        "sample_rate": sample_rate,
        "other_information": "Can also be present.",
    }

    with CWScope(**capture_info) as scope:
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


@patch.object(cw, "scope")
def test_with(mock_scope):
    mock_cwscope = MagicMock()
    mock_scope.return_value = mock_cwscope
    gain = 45
    samples = 7000
    offset = 0
    clock = 7372800
    sample_rate = "clkgen_x4"
    mock_cwscope.adc.oa.hwInfo.maxSamples.return_value = -1

    with CWScope(gain=gain,
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
