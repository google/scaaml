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
from copy import deepcopy
import time
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from chipwhisperer.common.utils import util
import numpy as np

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationError
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunication
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationSocket
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationVisa
from scaaml.capture.scope.lecroy.types import LECROY_CHANNEL_NAME_T, LECROY_COMMUNICATION_CLASS_NAME
from scaaml.capture.scope.scope_template import ScopeTemplate
from scaaml.io import Dataset


class LeCroy(AbstractSScope):
    """Scope context manager."""

    def __init__(self,
                 samples: int,
                 offset: int,
                 ip_address: str,
                 trace_channel: LECROY_CHANNEL_NAME_T,
                 trigger_channel: LECROY_CHANNEL_NAME_T,
                 communication_timeout: float,
                 trigger_timeout: float,
                 scope_setup_commands: List[Dict[str, Any]],
                 communication_class_name:
                 LECROY_COMMUNICATION_CLASS_NAME = "LeCroyCommunicationVisa",
                 **_) -> None:
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
          scope_setup_commands (List[Dict[str, Any]]): List of commands used
            to set up the scope. There are three possible actions taken in
            the order (command, method, query):

            - { "command": "command string" } The "command string" is sent to
              the scope after being formatted with
              `trace_channel=trace_channel, trigger_channel=trigger_channel`.

            - { "method": "method_name", "kwargs": {} } The method is called
              with given kwargs. The supported methods are: `set_trig_delay`.
              This allows to do settings which are dependent on other settings
              (e.g., setting TRIG_DELAY which depends on TIME_DIV) without
              having to change two parameters.

            - { "query": "query string" } The "query string" is sent to
              the scope after being formatted with
              `trace_channel=trace_channel, trigger_channel=trigger_channel`.
              A dictionary of {"query string": "answer"} for all of these can
              be obtained using `get_scope_answers`.

            For convenience the following commands are prepended:
              { "command": "COMM_HEADER OFF", },
              {  # Use full precision of measurements
                  "command": "COMM_FORMAT DEF9,WORD,BIN",
                  "query": "COMM_FORMAT?",
              },
              { "command": "TRMD SINGLE", },  # Trigger mode
              { "command": "AUTO_CALIBRATE OFF", },
              { "command": "OFFSET 0", },  # Center the trace vertically
            and the following is appended:
              {"command": "STOP"}  # Stop any signal acquisition

            For a description of possible commands and queries see
            https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf

          communication_class_name (LECROY_COMMUNICATION_CLASS_NAME): Which
            class to use for communication with the scope. Defaults to
            LeCroyCommunicationVisa, the other possibility is
            LeCroyCommunicationSocket.
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
        self._communication_class_name = communication_class_name

        self._scope_setup_commands = deepcopy(scope_setup_commands)

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
            communication_class_name=self._communication_class_name,
            scope_setup_commands=self._scope_setup_commands,
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

    def post_init(self, dataset: Dataset) -> None:
        """After initialization actions. Saves the scope identity information
        into the capture_info.
        """
        assert self._scope

        # Update capture info with oscilloscope details.
        dataset.capture_info.update(self._scope.get_identity_info())
        dataset.capture_info["scope_answers"] = self._scope.get_scope_answers()
        dataset.write_config()


class LeCroyScope(ScopeTemplate):
    """Scope."""

    def __init__(
            self, samples: int, offset: int, ip_address: str,
            trace_channel: LECROY_CHANNEL_NAME_T,
            trigger_channel: LECROY_CHANNEL_NAME_T,
            communication_timeout: float, trigger_timeout: float,
            scope_setup_commands: List[Dict[str, Any]],
            communication_class_name: LECROY_COMMUNICATION_CLASS_NAME) -> None:
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
          scope_setup_commands (List[Dict[str, Any]]): See docstring of
            `LeCroy`.
          communication_class_name (LECROY_COMMUNICATION_CLASS_NAME): Which
            class to use for communication with the scope.
        """
        self._samples = samples
        self._offset = offset
        self._ip_address = ip_address
        self._trace_channel = trace_channel
        self._trigger_channel = trigger_channel
        self._communication_timeout = communication_timeout
        self._trigger_timeout = trigger_timeout
        self._communication_class_name = communication_class_name

        # Trace and trigger
        self._last_trace = None

        # Scope object
        self._scope_communication: Optional[LeCroyCommunication] = None

        # Desired settings of the scope
        scope_setup_commands = deepcopy(scope_setup_commands)
        # Wrap default commands around the custom ones
        scope_setup_commands = [
            { "command": "COMM_HEADER OFF", },
            {  # Use full precision of measurements
                "command": "COMM_FORMAT DEF9,WORD,BIN",
                "query": "COMM_FORMAT?",
            },
            { "command": "TRMD SINGLE", },  # Trigger mode
            { "command": "AUTO_CALIBRATE OFF", },
            { "command": "OFFSET 0", },  # Center the trace vertically
        ] + scope_setup_commands + [  # Custom commands
            {"command": "STOP"}  # Stop any signal acquisition
        ]
        self._scope_setup_commands: Tuple[Dict[str, Any],
                                          ...] = tuple(scope_setup_commands)

        # Actual settings of the scope
        self._scope_answers: Dict[str, str] = {}

    def con(self, sn=None) -> bool:
        """Set the scope for capture."""
        communication_cls = {
            "LeCroyCommunicationVisa": LeCroyCommunicationVisa,
            "LeCroyCommunicationSocket": LeCroyCommunicationSocket,
        }[self._communication_class_name]
        # Connect to the oscilloscope.
        self._scope_communication = communication_cls(
            ip_address=self._ip_address,
            timeout=self._communication_timeout,
        )

        assert self._scope_communication is not None

        # Connect to the physical scope
        self._scope_communication.connect()

        # Run all setup commands
        for command in self._scope_setup_commands:
            self._run_command(command)

        # Success
        return True

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
            container=bytearray,
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

    def __repr__(self) -> str:
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

    def __str__(self) -> str:
        """Return string representation of self.
        """
        return self.__repr__()

    def get_identity_info(self) -> Dict[str, str]:
        """Get information about the oscilloscope identity.

        Returns: a dictionary containing the model, serial_number, and
        firmware_level (version).
        """
        assert self._scope_communication
        answer = self._scope_communication.query("*IDN?").rstrip()
        brand, model, serial_number, firmware_level = answer.split(",")
        assert brand == "LECROY"
        return {
            "lecroy_model": model,
            "lecroy_serial_number": serial_number,
            "lecroy_firmware_level": firmware_level,
        }

    def get_scope_answers(self) -> Dict[str, str]:
        """Return actual scope settings. When adding a scope setting the closes
        valid value is used or the setting is ignored. These are results of
        "query" in `scope_setup_commands` (see the init of `LeCroy`).
        """
        return deepcopy(self._scope_answers)

    def set_trig_delay(self, divs_left: float = -4.0) -> None:
        """Position the trigger point `divs_left` divisions left from the
        center of the screen. Defaults to showing one division before (10% of
        trace is pre-trigger).

        divs_left (float): How much to move the trigger point (in divisions to
          the left). Defaults to -4, which is 10% of trace is before the
          trigger.
        """
        assert self._scope_communication

        timebase = float(self._scope_communication.query("TIME_DIV?").strip())
        trig_position = divs_left * timebase
        self._scope_communication.write(f"TRIG_DELAY {trig_position}")

    def _run_command(self, setup_command: Dict[str, Any]) -> None:
        """For description see init of `LeCroy`.
        """
        assert self._scope_communication

        # Wildcards to fill in for commands
        wildcards = {
            "trace_channel": self._trace_channel,
            "trigger_channel": self._trigger_channel,
        }

        # First run a command
        if "command" in setup_command:
            command_string = setup_command["command"]
            # Fill in wildcards
            command_string = command_string.format(**wildcards)
            # Run the command
            self._scope_communication.write(command_string)

        # Run a method if there is one
        if "method" in setup_command:
            method_name = setup_command["method"]
            kwargs = setup_command.get("kwargs", {})
            if method_name == "set_trig_delay":
                self.set_trig_delay(**kwargs)

        # Update the actual setting
        if "query" in setup_command:
            query = setup_command["query"]
            query = query.format(**wildcards)
            self._scope_answers[query] = self._scope_communication.query(query)
