# Copyright 2025 Google LLC
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
"""Test PicoScope."""

import pytest
from typing import get_args

from picosdk.PicoDeviceEnums import picoEnum

from scaaml.capture.scope.picoscope import PicoScope


def test_resolution_type():
    defined = set(get_args(PicoScope.RESOLUTION_T))

    expected = set(picoEnum.PICO_DEVICE_RESOLUTION)
    assert defined == expected


def test_coveralls_happiness():
    with PicoScope(
            samples=1_000,
            sample_rate=5_000_000,
            offset=0,
            trace_channel="A",
            trace_probe_range=1.0,
            trace_coupling="DC50",
            trace_attenuation="1:1",
            trace_bw_limit="PICO_BW_FULL",
            trace_ignore_overflow=True,
            trigger_channel="PORT0",
            trigger_hysteresis="PICO_NORMAL_100MV",
            trigger_pin=1,
            trigger_range=5.0,
            trigger_level=1.5,
            trigger_coupling="AC",
            resolution="PICO_DR_8BIT",
    ) as pico_scope:
        pass
