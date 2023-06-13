# Copyright 2023 Google LLC
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

import base64
import time
from typing import Literal, Optional
from typing_extensions import TypeAlias
import xml.etree.ElementTree as ET

from chipwhisperer.common.utils import util
import numpy as np

from scaaml.capture.scope import AbstractSScope
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationError
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunication
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationVisa
from scaaml.capture.scope.scope_template import ScopeTemplate

LECROY_CHANNEL_NAME_T: TypeAlias = Literal["C1", "C2", "C3", "C4", "DIGITAL1",
                                           "DIGITAL2", "DIGITAL3", "DIGITAL4"]


class LeCroy(AbstractSScope):
    """Scope context manager."""

    def __init__(self, samples: int, offset: int, ip_address: str,
                 trace_channel: LECROY_CHANNEL_NAME_T,
                 trigger_channel: LECROY_CHANNEL_NAME_T,
                 communication_timeout: float, trigger_timeout: float, **_):
        """Create scope context.

        Args:
          samples (int): How many points to sample (length of the capture).
          offset (int): How many samples to discard.
          ip_address (str): IP address or hostname of the oscilloscope.
          trace_channel (LECROY_CHANNEL_NAME_T): Channel name.
          trigger_channel (LECROY_CHANNEL_NAME_T): Channel name.
          communication_timeout (float): Timeout communication after
            `communication_timeout` seconds.
          trigger_timeout (float): Number of seconds before the trigger times
            out (in seconds).
          _: LeCroy is expected to be initialized using capture_info
            dictionary, this parameter allows to have additional information
            there and initialize as LeCroy(**capture_info).
        """
        super().__init__(samples=samples, offset=offset)

        self._ip_address = ip_address
        self._trace_channel = trace_channel
        self._trigger_channel = trigger_channel
        self._communication_timeout = communication_timeout
        self._trigger_timeout = trigger_timeout

        # Check that the trace_channel is analog
        if not self._trace_channel.startswith("C"):
            raise ValueError(f"The trace channel should be analog, but is "
                             f"{self._trace_channel} instead")

        # Scope object
        self._scope: Optional[LeCroyScope] = None

    def __enter__(self):
        """Create scope context.

        Returns: self
        """
        # Do not allow nested with.
        assert self._scope is None
        self._scope = LeCroyScope(
            samples=self._samples,
            offset=self._offset,
            ip_address=self._ip_address,
            trace_channel=self._trace_channel,
            trigger_channel=self._trigger_channel,
            communication_timeout=self._communication_timeout,
            trigger_timeout=self._trigger_timeout,
        )
        assert self._scope is not None
        self._scope.con()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        del exc_type  # unused
        del exc_value  # unused
        del exc_tb  # unused

        if self._scope is None:
            return
        self._scope.dis()
        self._scope = None


class LeCroyScope(ScopeTemplate):
    """Scope."""

    def __init__(self, samples: int, offset: int, ip_address: str,
                 trace_channel: LECROY_CHANNEL_NAME_T,
                 trigger_channel: LECROY_CHANNEL_NAME_T,
                 communication_timeout: float, trigger_timeout: float):
        """Create scope context.

        Args:
          samples (int): How many points to sample (length of the capture).
          offset (int): How many samples to discard.
          ip_address (str): IP address or hostname of the oscilloscope.
          trace_channel (LECROY_CHANNEL_NAME_T): Channel name.
          trigger_channel (LECROY_CHANNEL_NAME_T): Channel name.
          communication_timeout (float): Timeout communication after
            `communication_timeout` seconds.
          trigger_timeout (float): Number of seconds before the trigger times
            out (in seconds).
        """
        self._samples = samples
        self._offset = offset
        self._ip_address = ip_address
        self._trace_channel = trace_channel
        self._trigger_channel = trigger_channel
        self._communication_timeout = communication_timeout
        self._trigger_timeout = trigger_timeout

        # Trace and trigger
        self._last_trace = None

        # Scope object
        self._scope_communication: Optional[LeCroyCommunication] = None

    def con(self, sn=None) -> bool:
        """Set the scope for capture."""
        # Connect to the oscilloscope.
        self._scope_communication = LeCroyCommunicationVisa(
            ip_address=self._ip_address,
            timeout=self._communication_timeout,
        )

        assert self._scope_communication is not None

        # Scope settings
        self._scope_communication.connect()
        self._scope_communication.write("COMM_HEADER OFF")
        self._scope_communication.write("COMM_FORMAT DEF9,WORD,BIN")
        self._scope_communication.write("TRMD SINGLE")
        self._scope_communication.write("STOP")
        return True  # Success

    def dis(self) -> bool:
        """Disconnect from the scope."""
        assert self._scope_communication is not None
        self._scope_communication.close()
        self._scope_communication = None
        return True  # Success

    def arm(self):
        """Prepare the scope for capturing."""
        assert self._scope_communication is not None
        self._scope_communication.write("ARM")
        arm_result = self._scope_communication.query("TRMD?")
        if arm_result != "SINGLE":
            raise ValueError(f"TRMD result must be SINGLE, got {arm_result}")

    def capture(self, poll_done: bool = False) -> bool:
        """Capture one trace and return True if timeout has happened
        (possible capture failure).

        Args:
          poll_done: Not supported in LeCroy, but a part of API.

        Returns: True if the trace needs to be recaptured due to timeout.
        """
        del poll_done  # unused

        assert self._scope_communication is not None

        try:
            # Wait for trigger
            for _ in range(10):
                time.sleep(self._trigger_timeout / 10)
                status = self._scope_communication.query("TRMD?")
                if status == "STOP":
                    break
            assert self._scope_communication.query("TRMD?") == "STOP"

            # Get trace
            wave = self._scope_communication.get_waveform(
                channel=self._trace_channel)
            self._last_trace = wave.get_wave1(
                first_sample=self._offset,
                length=self._samples,
            )

            return False  # No need to recapture.
        except LeCroyCommunicationError:
            # Reconnect and return True (need to be recaptured).
            self.dis()
            time.sleep(1)
            self.con()
            return True

    def get_last_trace(self, as_int: bool = False) -> np.ndarray:
        """Return a copy of the last trace.

        Args:
          as_int (bool): Not supported, part of CW API. Return integer
            representation of the trace. Defaults to False meaning return
            np.array of dtype np.float32.

        Returns: np array representing the last captured trace.
        """
        if self._last_trace is None:
            raise RuntimeError("No trace has been captured so far.")

        if as_int:
            msg = "Returning trace as integers is not implemented."
            raise NotImplementedError(msg)

        return self._last_trace

    def get_last_trigger_trace(self) -> np.ndarray:
        """Return a copy of the last trigger trace."""
        assert self._scope_communication is not None

        if self._trigger_channel.startswith("C"):
            # Get analog trigger:
            waveform = self._scope_communication.get_waveform(
                channel=self._trigger_channel)
            return waveform.get_wave1(
                first_sample=self._offset,
                length=self._samples,
            )

        # Get digital trigger:
        trigger = self._scope_communication.query_binary_values(
            f"{self._trigger_channel}:WF?",
            datatype="B",
        )

        root = ET.fromstring(bytes(trigger).decode("ascii"))
        # Find the binary data (BinaryData is present)
        binary_data = root.findall(".//BinaryData")[0].text
        if binary_data is not None:
            decoded = base64.b64decode(binary_data)
        else:
            # No trigger array
            return np.array([], dtype=np.uint8)

        np_trigger = np.frombuffer(decoded, dtype=np.uint8)
        return np_trigger

    def __str__(self) -> str:
        """Return string representation of self.
        """
        return util.dict_to_str({
            "samples": self._samples,
            "offset": self._offset,
            "ip_address": self._ip_address,
            "trace_channel": self._trace_channel,
            "trigger_channel": self._trigger_channel,
            "communication_timeout": self._communication_timeout,
            "trigger_timeout": self._trigger_timeout,
        })
