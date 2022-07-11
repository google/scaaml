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
"""Test Pico6424E."""

from decimal import Decimal
import pytest

from scaaml.capture.scope.ps6424e import Pico6424E


def test_get_timebase_too_large():
    """Test sample rate to timebase conversion for too large sampling rate."""
    # Test too large
    with pytest.raises(ValueError) as value_error:
        timebase = Pico6424E._get_timebase(1 + 5e9)
    expected_msg: str = "This scope supports at most 5GHz sample_rate."
    got_msg: str = str(value_error.value)
    assert expected_msg == got_msg


def test_get_timebase_small_timebase():
    """Test sample rate to timebase conversion for large sampling rates."""
    table = {
        # sample rate [Hz]: timebase as for PicoScope
        5e9: 0,
        2.5e9: 1,
        1.25e9: 2,
        625e6: 3,
        312.5e6: 4,
    }

    # Test exact inputs
    for sample_rate, timebase in table.items():
        assert Pico6424E._get_timebase(sample_rate).value == timebase

    # Test a bit distorted inputs (choose a bit faster sampling rate)
    for sample_rate, timebase in table.items():
        assert Pico6424E._get_timebase(sample_rate - 1e6).value == timebase


def test_get_timebase_large_timebase():
    """Test sample rate to timebase conversion for smaller sampling rates."""
    # Test smaller values
    for timebase in range(5, 1_000):
        sample_rate = Decimal(156_250_000) / (timebase - 4)
        assert Pico6424E._get_timebase(sample_rate).value == timebase

    # Test larger values
    for timebase in range(1_000, 1_000_000, 1_000):
        sample_rate = Decimal(156_250_000) / (timebase - 4)
        assert Pico6424E._get_timebase(sample_rate).value == timebase
