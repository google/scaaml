# Copyright 2021-2024 Google LLC
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

from types import TracebackType
from typing import Any, Literal, Optional
from typing_extensions import Self, TypeAlias

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.scope_template import ScopeTemplate
from scaaml.capture.scope.ps6424e import Pico6424E as PicoScope6424E


class PicoScope(AbstractSScope):
    """Scope context manager."""
    CHANNEL_T: TypeAlias = Literal["A", "B", "C", "D", "E", "F", "G", "H",
                                   "PORT0", "PORT1"]
    COUPLING_T: TypeAlias = Literal["AC", "DC", "DC50"]
    ATTENUATION_T: TypeAlias = Literal["1:1", "1:10"]
    BW_LIMIT_T: TypeAlias = Literal["PICO_BW_FULL", "PICO_BW_20MHZ",
                                    "PICO_BW_200MHZ"]

    def __init__(self, *, samples: int, sample_rate: float, offset: int,
                 trace_channel: CHANNEL_T, trace_probe_range: float,
                 trace_coupling: COUPLING_T, trace_attenuation: ATTENUATION_T,
                 trace_bw_limit: BW_LIMIT_T, trace_ignore_overflow: bool,
                 trigger_channel: CHANNEL_T, trigger_hysteresis: Optional[str],
                 trigger_pin: Optional[int], trigger_range: float,
                 trigger_level: float, trigger_coupling: COUPLING_T,
                 **_: Any) -> None:
        """Create scope context.

        Args:
          samples (int): How many points to sample (length of the capture).
          sample_rate (float): How many samples per second to take in Hz. The
            setting will be the closest PicoScope sampling rate that is at
            least sample_rate.
          offset (int): How many samples are discarded between the trigger
            event and the start of the trace.
          trace_channel (str): Which channel to use for the signal.
          trace_probe_range (float): Should be in [0.02, 0.05, 0.1, 0.2, 0.5,
            1.0, 2.0, 5.0, 10.0, 20.0, 50.0] (in V).
          trace_coupling (str): Which coupling to use, one of AC, DC, DC50 (DC
            with 50 Ohm).
          trace_attenuation (str): Which attenuation to use, either 1:1 or 1:10.
            For BNC-SMA pigtail use "1:1", for oscilloscope passive probe
            use "1:10".
          trace_bw_limit (str): Bandwidth limit. Filter out too high
            frequencies (can prevent overflows in trace).
          trace_ignore_overflow (bool): If True then we ignore trace overflow.
            Otherwise PicoScope6424E.capture returns True (timeouts).
          trigger_channel (str): Which channel to use. Values are: "A", "B",
            "C", "D", (if the scope has those "E", "F", "G", "H"). If the
            scope has digital ports then "PORT0" and "PORT1".
          trigger_hysteresis (Optional[str]): None if the trigger is analog.
            Otherwise one of PICO_VERY_HIGH_400MV, PICO_HIGH_200MV,
            PICO_NORMAL_100MV, PICO_LOW_50MV.
          trigger_pin (Optional[int]): Only when using MSO port as
            trigger_channel. Which pin to trigger on.
          trigger_range (float): Should be in [0.02, 0.05, 0.1, 0.2, 0.5,
            1.0, 2.0, 5.0, 10.0, 20.0, 50.0] (in V). Analog trigger only.
          trigger_level (float): When to trigger (in V).
          trigger_coupling (str): Which coupling to use, one of AC, DC, DC50
            (DC with 50 Ohm). Analog trigger only.
          _: PicoScope is expected to be initialized using capture_info
            dictionary, this parameter allows to have additional information
            there and initialize as PicoScope(**capture_info).
        """
        super().__init__(samples=samples, offset=offset)
        self._sample_rate = sample_rate

        # Trace settings
        self._trace_channel: PicoScope.CHANNEL_T = trace_channel
        self._trace_probe_range: float = trace_probe_range
        self._trace_coupling: PicoScope.COUPLING_T = trace_coupling
        self._trace_attenuation: PicoScope.ATTENUATION_T = trace_attenuation
        self._trace_bw_limit: PicoScope.BW_LIMIT_T = trace_bw_limit
        self._ignore_overflow: bool = trace_ignore_overflow

        # Trigger settings
        self._trigger_channel: PicoScope.CHANNEL_T = trigger_channel
        self._trigger_pin: Optional[int] = trigger_pin
        self._trigger_hysteresis: Optional[str] = trigger_hysteresis
        self._trigger_range: float = trigger_range
        self._trigger_level: float = trigger_level
        self._trigger_coupling: PicoScope.COUPLING_T = trigger_coupling

        # Scope object
        self._scope: Optional[ScopeTemplate] = None

    def __enter__(self) -> Self:  # pragma: no cover
        """Create scope context.

        Suppose that the signal is channel A and the trigger is channel B.

        Returns: self
        """
        # Do not allow nested with.
        assert self._scope is None

        # Connect to the oscilloscope.
        self._scope = PicoScope6424E()
        self._scope.con()

        # Trace channel settings.
        self._scope.trace.channel = self._trace_channel
        # pylint: disable=line-too-long
        self._scope.trace.range = self._trace_probe_range  # type: ignore[assignment]
        # pylint: enable=line-too-long
        self._scope.trace.coupling = self._trace_coupling
        self._scope.trace.probe_attenuation = self._trace_attenuation
        self._scope.trace.bw_limit = self._trace_bw_limit
        self._scope.ignore_overflow = self._ignore_overflow

        # Trigger settings.
        self._scope.trigger.channel = self._trigger_channel
        self._scope.trigger.port_pin = self._trigger_pin
        self._scope.trigger.hysteresis = self._trigger_hysteresis
        self._scope.trigger.trigger_level = self._trigger_level
        self._scope.trigger.coupling = self._trigger_coupling
        self._scope.trigger.probe_attenuation = "1:1"
        # pylint: disable=line-too-long
        self._scope.trigger.range = self._trigger_range  # type: ignore[assignment]
        # pylint: enable=line-too-long

        # Number of samples settings.
        self._scope.sample_rate = self._sample_rate
        self._scope.sample_length = self._samples
        self._scope.sample_offset = self._offset

        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:  # pragma: no cover
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        if self._scope is None:
            return
        self._scope.dis()
        self._scope = None
