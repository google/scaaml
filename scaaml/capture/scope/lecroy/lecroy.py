# Copyright 2023-2024 Google LLC
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
import logging
from pathlib import Path
import time

from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Self

import xml.etree.ElementTree as ET

from chipwhisperer.common.utils import util
import numpy as np

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationError
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunication
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationSocket
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationVisa
from scaaml.capture.scope.lecroy.lecroy_waveform import DigitalChannelWaveform
from scaaml.capture.scope.lecroy.types import (
    LECROY_CAPTURE_AREA,
    LECROY_CHANNEL_NAME_T,
    LECROY_COMMUNICATION_CLASS_NAME,
    LECROY_DIG_LINE_T,
)
from scaaml.capture.scope.scope_template import ScopeTemplate, ScopeTraceType, ScopeTriggerTraceType
from scaaml.io import Dataset


class LeCroy(AbstractSScope):
    """Scope context manager."""

    def __init__(self,
                 *,
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
                 trigger_line: LECROY_DIG_LINE_T | None = None,
                 **_: Any) -> None:
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
              the scope after being formatted with the following wildcards:
              - `trace_channel=trace_channel`,
              - `trigger_channel=trigger_channel`,
              - `trigger_line=trigger_line`.
              - `samples=samples`.

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
              { "command": "{trace_channel}:OFFSET 0", },  # Center the trace vertically
            and the following is appended:
              {"command": "STOP"}  # Stop any signal acquisition

            For a description of possible commands and queries see
            https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf

          communication_class_name (LECROY_COMMUNICATION_CLASS_NAME): Which
          class to use for communication with the scope. Defaults to
          LeCroyCommunicationVisa, the other possibility is
          LeCroyCommunicationSocket.

          trigger_line (LECROY_DIG_LINE_T | None): If trigger channel is
          digital (in LECROY_DIG_CHANNEL_NAME_T) then one also needs to specify
          which line ("D0", "D1", ... "D15" if there are 16 lines). Checked
          against `trigger_channel`.

          _: LeCroy is expected to be initialized using capture_info
          dictionary, this parameter allows to have additional information
          there and initialize as LeCroy(**capture_info).
        """
        super().__init__(samples=samples, offset=offset)

        self._ip_address = ip_address
        self._trace_channel: LECROY_CHANNEL_NAME_T = trace_channel
        self._trigger_channel: LECROY_CHANNEL_NAME_T = trigger_channel

        if trigger_line is None and trigger_channel.startswith("D"):
            raise ValueError(f"No digital line selected")

        self._trigger_line: LECROY_DIG_LINE_T | None = trigger_line
        self._communication_timeout = communication_timeout
        self._trigger_timeout = trigger_timeout
        # pylint:disable=C0301
        self._communication_class_name: LECROY_COMMUNICATION_CLASS_NAME = communication_class_name

        self._scope_setup_commands = deepcopy(scope_setup_commands)

        # Check that the trace_channel is analog
        if not self._trace_channel.startswith("C"):
            raise ValueError(f"The trace channel should be analog, but is "
                             f"{self._trace_channel} instead")

        # Scope object
        self._scope = None

    def __enter__(self) -> Self:
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
            trigger_line=self._trigger_line,
        )
        assert self._scope is not None
        self._scope.con()
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
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

    @property
    def scope(self) -> ScopeTemplate:
        assert self._scope
        return self._scope

    def post_init(self, dataset: Dataset) -> None:
        """After initialization actions. Saves the scope identity information
        into the capture_info.
        """
        assert isinstance(self._scope, LeCroyScope)

        # Update capture info with oscilloscope details.
        dataset.capture_info.update(self._scope.get_identity_info())
        dataset.capture_info["scope_answers"] = self._scope.get_scope_answers()
        dataset.write_config()

    def print_screen(
            self,
            file_path: Path,
            capture_area: LECROY_CAPTURE_AREA = "GRIDAREAONLY") -> None:
        """Take a print screen and transfer it to this computer.

        Args:

          file_path (Path): Where to save the print screen file.

          capture_area (LECROY_CAPTURE_AREA): Capture the trace area, full
          window, or full screen. Defaults to the trace area.
        """
        assert isinstance(self._scope, LeCroyScope)
        source_file_path: str = self._scope.call_print_screen(
            capture_area=capture_area)
        self._scope.retrieve_file(source_file_path=source_file_path,
                                  destination_file_path=file_path)
        self._scope.delete_file(file_path=source_file_path)


class LeCroyScope(ScopeTemplate):
    """Scope."""

    def __init__(
        self,
        *,
        samples: int,
        offset: int,
        ip_address: str,
        trace_channel: LECROY_CHANNEL_NAME_T,
        trigger_channel: LECROY_CHANNEL_NAME_T,
        communication_timeout: float,
        trigger_timeout: float,
        scope_setup_commands: List[Dict[str, Any]],
        communication_class_name: LECROY_COMMUNICATION_CLASS_NAME,
        trigger_line: LECROY_DIG_LINE_T | None,
    ) -> None:
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

          trigger_line (LECROY_DIG_LINE_T | None): If trigger channel is
          digital (in LECROY_DIG_CHANNEL_NAME_T) then one also needs to specify
          which line ("D0", "D1", ... "D15" if there are 16 lines). Checked
          against `trigger_channel`.
        """
        self._samples = samples
        self._offset = offset
        self._ip_address = ip_address
        self._trace_channel: LECROY_CHANNEL_NAME_T = trace_channel
        self._trigger_channel: LECROY_CHANNEL_NAME_T = trigger_channel
        self._trigger_line: LECROY_DIG_LINE_T | None = trigger_line
        self._communication_timeout = communication_timeout
        self._trigger_timeout = trigger_timeout
        self._communication_class_name = communication_class_name

        # Trace and trigger
        self._last_trace: Optional[ScopeTraceType] = None

        # Scope object
        self._scope_communication: Optional[LeCroyCommunication] = None

        # Desired settings of the scope
        # Wrap default commands around the custom ones
        commands = [
            { "command": "COMM_HEADER OFF", },
            {  # Use full precision of measurements
                "command": "COMM_FORMAT DEF9,WORD,BIN",
                "query": "COMM_FORMAT?",
            },
            { "command": "TRMD SINGLE", },  # Trigger mode
            { "command": "AUTO_CALIBRATE OFF", },
            { "command": "{trace_channel}:OFFSET 0", },  # Center the trace vertically
            {
                "command": f"WAVEFORM_SETUP SP,1,NP,{self._samples},FP,{self._offset},SN,0",
                "query": "WAVEFORM_SETUP?",
            }
        ]
        commands.extend(deepcopy(scope_setup_commands))  # Custom commands
        commands.append({"command": "STOP"})  # Stop any signal acquisition
        self._scope_setup_commands: Tuple[Dict[str, Any], ...] = tuple(commands)

        # Actual settings of the scope
        self._scope_answers: Dict[str, str] = {}

        # When using digital trigger then we need to interpolate it (since the
        # sampling rate might be different, we might be cutting the trace,
        # sparsing, ...).
        self._last_trace_wave: LecroyWaveform | None = None

        self._logger = logging.getLogger("scaaml.capture.scope.lecroy."
                                         "lecroy_scope")

    def con(self,
            sn: Optional[str] = None,
            sleep_after_command: float = 1.0) -> bool:
        """Set the scope for capture.

        Args:

          sn (str | None): The serial number of the probe. Ignored for LeCroy
          scope (but left for compatibility with ChipWhisperer. Defaults to
          None. Use `ip_address` instead to identify the correct oscilloscope.
          Technically it is possible to check the serial number but listing all
          available oscilloscopes is not supported by our library.

          sleep_after_command (float): Sleep interval length in seconds after
          each command (see `scope_setup_commands`). This is to let the
          physical changes to happen. Defaults to 1.0 second.
        """
        del sn  # unused

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
            time.sleep(sleep_after_command)

        # Success
        return True

    def dis(self) -> bool:
        """Disconnect from the scope."""
        assert self._scope_communication is not None
        self._scope_communication.close()
        self._scope_communication = None
        return True  # Success

    def arm(self) -> None:
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
            trigger_mode: str = self._scope_communication.query("TRMD?")
            if trigger_mode != "STOP":
                self.logger.error(
                    f"Expected STOP, got %s probably a timeout",
                    trigger_mode,
                )
                return True

            # Get trace
            wave = self._scope_communication.get_waveform(
                channel=self._trace_channel,)
            self._last_trace = wave.get_wave1(
                first_sample=0,  # This was set by WAVEFORM_SETUP
                length=-1,  # This was set by WAVEFORM_SETUP
            )
            self._last_trace_wave = wave

            return False  # No need to recapture.
        except LeCroyCommunicationError:
            # Reconnect and return True (need to be recaptured).
            self._last_trace_wave = None
            self.dis()
            time.sleep(1)
            self.con()
            return True

    def get_last_trace(self, as_int: bool = False) -> ScopeTraceType:
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

    def get_last_trigger_trace(self) -> ScopeTriggerTraceType:
        """Return a copy of the last trigger trace."""
        assert self._scope_communication is not None

        if self._trigger_channel.startswith("C"):
            # Get analog trigger:
            waveform = self._scope_communication.get_waveform(
                channel=self._trigger_channel,)
            return waveform.get_wave1(
                first_sample=0,  # This was set by WAVEFORM_SETUP
                length=-1,  # This was set by WAVEFORM_SETUP
            )

        # Return digital trigger wave.
        xml_data: str = self._scope_communication.get_xml_dig_data(
            channel=self._trigger_channel)
        digital_wave = DigitalChannelWaveform(xml_data=xml_data)

        # Interpolation to match digital wave time with analog (sampling rates
        # could be different, ...).
        assert self._trigger_line is not None
        trigger = digital_wave.traces[self._trigger_line]

        # Interpolate to full trace length.
        assert self._last_trace_wave
        wave_description = self._last_trace_wave._wave_description

        # Only supported options.
        assert wave_description.segment_index == 0
        assert wave_description.subarray_count == 1
        assert wave_description.sweeps_per_acq == 1
        assert wave_description.record_type == "single_sweep"
        assert wave_description.ris_sweeps == 1

        # Full trace is wave_array_count, but we do FP and SP:
        trace_stretch = wave_description.points_per_screen
        trigger = np.interp(
            x=np.arange(trace_stretch) / trace_stretch,  # Resulting times
            xp=np.arange(len(trigger)) / len(trigger),  # We have
            fp=trigger,
        )

        # These values could have been changed by running hard-coded
        # WAVEFORM_SETUP command and setting different samples / offset.
        if self._samples != wave_description.wave_array_count:
            self._logger.error(
                "The expected number of samples %s != the actual number %s",
                self._samples,
                wave_description.wave_array_count,
            )
        if self._offset != wave_description.first_point:
            self._logger.error(
                "The expected offset %s != the actual offset %s",
                self._offset,
                wave_description.first_point,
            )

        # Cut the correct part out of the whole screen trigger line signal. We
        # could use self._offset and self._samples but those might not be up to
        # date -> prefer to use parsed values.
        trigger = trigger[wave_description.first_point::wave_description.
                          sparsing_factor][:wave_description.wave_array_count]
        # Digital trigger is expected to be binary but we have interpolated.
        trigger = np.round(trigger)

        return trigger

    def __repr__(self) -> str:
        """Return string representation of self.
        """
        val = util.dict_to_str({
            "samples": self._samples,
            "offset": self._offset,
            "ip_address": self._ip_address,
            "trace_channel": self._trace_channel,
            "trigger_channel": self._trigger_channel,
            "communication_timeout": self._communication_timeout,
            "trigger_timeout": self._trigger_timeout,
        })  # type: ignore[no-untyped-call]
        return str(val)

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
            "trigger_line": self._trigger_line,
            "samples": self._samples,
            "offset": self._offset,
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

    def retrieve_file(self, source_file_path: str,
                      destination_file_path: Path) -> int:
        r"""Transfer a file and return how many bytes were transferred.

        Args:
          source_file_path (str): The file path path on the oscilloscope, e.g.,
            'D:\PRINT-SCREEN--00001.PNG'.
          destination_file_path (Path): The path to the file being saved. Parent
            directories must exist.

        Returns: the number of transferred bytes.
        """
        assert self._scope_communication

        # Query to get file data from the scope.
        # Answer format:
        # Beware that the #9 (the nine is base 16).
        # TRFL #9nnnnnnnnn<data><crc>
        # Where the TRFL is omitted due to setting "COMM_HEADER OFF"
        # nnnnnnnnn is the file size (bytes in ASCII for decimal
        #   representation) in bytes
        # crc is the 32-bit CRC plus 8-byte CRC trailer
        ans = self._scope_communication.query_binary_values(
            f"TRANSFER_FILE? DISK,HDD,FILE,'{source_file_path}'",
            container=bytearray,
        )

        # With socket we need to strip the header (if there is no header we are
        # using pyvisa).
        strip_header = True
        file_size: int = 0
        # Check for start: either "TRFL #9" or "#9"
        if ans[:2] == b"#9":
            # Just number of bytes and CRC
            ans = ans[2:]
        elif ans[:7] == b"TRFL #9":
            # With header, number of bytes, and CRC
            ans = ans[7:]
        else:
            # pyvisa automatically strips this header
            strip_header = False
            file_size = len(ans)

        if strip_header:
            # Decode file_size from bytes of ASCII
            file_size = int(ans[:9].decode("ascii"))
            ans = ans[9:]  # drop the file_size

            # TODO check CRC

        # Write the file
        with open(destination_file_path, "wb") as output_file:
            output_file.write(ans[:file_size])

        return file_size

    def call_print_screen(self, capture_area: LECROY_CAPTURE_AREA) -> str:
        """Set the HARDCOPY_SETUP variable and call SCREEN_DUMP.

        Args:
          capture_area (LECROY_CAPTURE_AREA): Capture the whole screen
            (resp., window) or just the grid with trace.

        Returns: The file path on the scope.
        """
        assert self._scope_communication

        # Setup where to save the file
        self._scope_communication.write(
            f"HARDCOPY_SETUP DEV,PNG,FORMAT,LANDSCAPE,BCKG,BLACK,DEST,FILE,DIR,"
            f"\"D:\",AREA,{capture_area},FILE,\"PRINT-SCREEN.PNG\"")
        hardcopy_setup_answer = self._scope_communication.query(
            "HARDCOPY_SETUP?")
        self._scope_communication.write("SCREEN_DUMP")

        # Parse the full DOS path to the print screen file
        hardcopy_setup = hardcopy_setup_answer.split(",")[8:]
        assert hardcopy_setup[0] == "DIR"
        # HARDCOPY_SETUP DEV,<device>,FORMAT,<format>,BCKG,<bckg>,
        # DEST,<destination>,DIR,"<directory>",AREA,<hardcopyarea>
        # [,FILE,"<filename>",PRINTER,"<printername>",PORT,<portname>]
        # <device>:= {BMP, JPEG, PNG, TIFF}
        # <format>:= {PORTRAIT, LANDSCAPE}
        # <bckg>:= {BLACK, WHITE}
        # <destination>:= {PRINTER, CLIPBOARD, EMAIL, FILE, REMOTE}
        # <area>:= {GRIDAREAONLY, DSOWINDOW, FULLSCREEN}
        # <directory>:= legal DOS path, for FILE mode only
        # <filename>:= filename string, no extension, for FILE mode only
        # <printername>:= valid printer name, for PRINTER mode only
        # <portname>:= {GPIB, NET}
        dir_i = hardcopy_setup.index("DIR")
        # FILE is there twice (FILE,DIR and FILE,"<FILENAME>")
        file_i = hardcopy_setup.index("FILE")
        directory = hardcopy_setup[dir_i + 1].strip("\"'")
        file = hardcopy_setup[file_i + 1].strip("\"'")
        file_name = directory + "\\" + file

        return file_name

    def delete_file(self, file_path: str) -> None:
        """Delete a given file on the scope.

        Args:
          file_path (str): The path to the file.
        """
        assert self._scope_communication

        # Delete the file
        self._scope_communication.write(
            f"DELETE_FILE DISK,HDD,FILE,'{file_path}'")
