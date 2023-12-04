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

import numpy as np

from scaaml.capture.scope.lecroy.lecroy_waveform import LecroyWaveform


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
