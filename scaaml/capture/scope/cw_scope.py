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
from typing import Any, Optional, Type
from typing_extensions import Self

import chipwhisperer as cw
from chipwhisperer.capture.scopes.cwnano import CWNano

from scaaml.capture.scope.scope_base import AbstractSScope


class CWScope(AbstractSScope):
    """Scope context manager."""

    def __init__(self,
                 *,
                 gain: int,
                 samples: int,
                 offset: int,
                 clock: int,
                 sample_rate: str,
                 cw_scope_serial_number: Optional[str] = None,
                 **_: Any) -> None:
        """Create scope context.

        Args:
          gain: Gain of the scope.
          samples: How many points to sample (length of the capture).
          offset: Number of samples to wait before starting recording data.
          clock: CLKGEN output frequency (in Hz).
          sample_rate: Clock source for cw.ClockSettings.adc_src.
          cw_scope_serial_number (Optional[str]): Serial number of the
            ChipWhisperer. Can be obtained by calling cw.list_devices() (the
            key "sn"). Defaults to None -- used when there is only one device.
          _: CWScope is expected to be initialized using the capture_info
            dictionary which may contain extra keys (additional information
            about the capture; the capture_info dictionary is saved in the
            info file of the dataset). Thus we can ignore the rest of keyword
            arguments.

        Expected use:
          capture_info = {
              'gain': gain,
              'samples': samples,
              'offset': offset,
              'clock': clock,
              'sample_rate': sample_rate,
              'other_information': 'Can also be present.',
          }
          with CWScope(**capture_info) as scope:
              # Use the scope object.
        """
        super().__init__(samples=samples, offset=offset)

        self._gain = gain
        self._clock = clock
        self._sample_rate = sample_rate
        self._basic_mode = "rising_edge"
        self._triggers = "tio4"
        self._freq_ctr_src = "clkgen"
        self._presamples = 0
        self._scope = None
        self._sn = cw_scope_serial_number

    def __enter__(self) -> Self:
        """Create scope context.

        Returns: self
        """
        assert self._scope is None  # Do not allow nested with.

        scope = cw.scope(sn=self._sn)
        assert not isinstance(scope, CWNano)
        self._scope = scope

        self._scope.gain.db = self._gain
        assert self._scope.adc.oa.hwInfo
        max_samples = self._scope.adc.oa.hwInfo.maxSamples()
        if (self._samples > max_samples and
                self._scope.adc.oa.hwInfo.is_cw1200()):
            self._scope.adc.stream_mode = True
        self._scope.adc.samples = self._samples
        self._scope.adc.offset = self._offset
        self._scope.adc.basic_mode = self._basic_mode
        self._scope.clock.clkgen_freq = self._clock
        self._scope.trigger.triggers = self._triggers
        self._scope.clock.adc_src = self._sample_rate
        self._scope.clock.freq_ctr_src = self._freq_ctr_src
        self._scope.adc.presamples = self._presamples
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        if self._scope is None:  # pragma: no cover
            return
        self._scope.dis()
        self._scope = None
