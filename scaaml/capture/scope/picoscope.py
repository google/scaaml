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

"""Context manager for the scope."""

import chipwhisperer as cw

from scaaml.capture.scope import AbstractSScope
from scaaml.capture.scope.ps6424e import Pico6424E as PicoScope6424E


class PicoScope(AbstractSScope):
    """Scope context manager."""
    def __init__(self,
                 samples: int,
                 trigger_level: float,
                 trigger_range: float,
                 sample_rate: float,
                 offset: int,
                 trace_probe_range: float,
                 **_):
        """Create scope context.

        Args:
          samples:
          trigger_level:
          trigger_range:
          offset:
          sample_rate (float): How many samples per second to take in Hz. The
            setting will be the closest PicoScope sampling rate that is at
            least sample_rate.
          trace_probe_range: [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] in V
        """
        super().__init__(samples=samples, offset=offset)
        self._sample_rate = sample_rate
        self._trigger_range = trigger_range
        self._trace_probe_range = trace_probe_range
        self._trigger_level = trigger_level
        self._scope = None

    def __enter__(self):
        """Create scope context.

        Suppose that the signal is channel A and the trigger is channel B.

        Returns: self
        """
        # Do not allow nested with.
        assert self._scope is None

        # Connect to the oscilloscope.
        picoscope = PicoScope6424E()
        picoscope.con()

        # Trace channel settings.
        picoscope.trace.range = self._trace_probe_range
        picoscope.trace.channel = "A"
        picoscope.trace.coupling = "AC"
        picoscope.trace.range = self._trace_probe_range

        # For BNC-SMA pigtail:
        picoscope.trace.probe_attenuation = "1:1"
        # For oscilloscope passive probe:
        #picoscope.trace.probe_attenuation = "1:10"

        # Trigger settings.
        picoscope.trigger.channel = "B"
        picoscope.trigger.trigger_level = self._trigger_level
        picoscope.trigger.coupling = "DC"
        picoscope.trigger.probe_attenuation = "1:1"
        picoscope.trigger.range = self._trigger_range

        # Number of samples settings.
        picoscope.sample_rate = self._sample_rate
        picoscope.sample_length = self._samples
        picoscope.sample_offset = self._offset

        self._scope = picoscope
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        self._scope.dis()
        self._scope = None

    @property
    def scope(self):
        """Returns the scope object."""
        return self._scope
