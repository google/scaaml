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
"""Test LecroyWaveform."""

import gzip

import numpy as np

from scaaml.capture.scope.lecroy.lecroy import LeCroyScope
from scaaml.capture.scope.lecroy.lecroy_waveform import LecroyWaveform
from scaaml.capture.scope.lecroy.lecroy_waveform import DigitalChannelWaveform


def test_parse_waveform():
    with open("tests/capture/scope/lecroy/raw_waveform.bin", "rb") as f:
        raw_waveform = f.read()
    lw = LecroyWaveform(raw_waveform)

    assert lw._unit == "h"  # reading word values

    expected = np.load("tests/capture/scope/lecroy/wave_1.npy",
                       allow_pickle=False)

    assert all(lw.get_wave1() == expected)


def test_parse_waveform_byte():
    with open("tests/capture/scope/lecroy/raw_waveform_byte.bin", "rb") as f:
        raw_waveform = f.read()
    lw = LecroyWaveform(raw_waveform)

    assert lw._unit == "b"  # reading byte values

    expected = np.load("tests/capture/scope/lecroy/wave_1_byte.npy",
                       allow_pickle=False)

    assert all(lw.get_wave1() == expected)


def test_parse_digital_waveform():
    # Trace and trigger were both connected to the calibration output of the
    # oscilloscope.

    test_dir = "tests/capture/scope/lecroy/"
    with open(
        (test_dir + "lecroy_samples_5432_offset_3141_trig_delay_2.3_SP_7123_" +
         "time_div_0.5MS_lecroy_raw_data_waveform.bin"), "rb") as f:
        raw_waveform = f.read()
    lw = LecroyWaveform(raw_waveform)

    with gzip.open(
        (test_dir + "lecroy_samples_5432_offset_3141_trig_delay_2.3_SP_7123_" +
         "time_div_0.5MS_lecroy_digital_waveform.xml.gz"), "rb") as f:
        xml_data = str(f.read(), encoding="ascii")
        digital_wave = DigitalChannelWaveform(xml_data=xml_data)

    assert lw._unit == "h"  # reading byte values
    assert len(digital_wave.traces) == 3
    assert digital_wave.generic_info["Source"] == "DIGITAL1"
    assert digital_wave.horizontal_unit == "S"
    assert digital_wave.horizontal_resolution == 1e-12
    assert digital_wave.sampling_rate == 1 / digital_wave.horizontal_per_step
    assert len(digital_wave.trace_infos) == len(digital_wave.traces)

    trace = lw.get_wave1(
        first_sample=0,  # This was set by WAVEFORM_SETUP
        length=-1,  # This was set by WAVEFORM_SETUP
    )
    trace_high = trace > 0.5

    #trigger = digital_wave.traces["D2"]
    oscilloscope = LeCroyScope(
        samples=len(trace),
        offset=lw._wave_description.first_point,
        ip_address="192.128.0.0",
        trace_channel="C3",
        trigger_channel="DIGITAL1",
        communication_timeout=1.0,
        trigger_timeout=1.0,
        scope_setup_commands=[],
        communication_class_name="LeCroyCommunicationSocket",
        trigger_line="D2",
    )

    # fool the scope that it captured
    oscilloscope._last_trace = trace
    oscilloscope._last_trace_wave = lw

    trigger = oscilloscope._compute_digital_trigger(xml_data=xml_data)

    difference = int(np.sum(np.logical_xor(trace_high, trigger)))
    assert difference <= 1
