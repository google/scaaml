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
from pathlib import Path
import time
from typing import Dict, Optional
import xml.etree.ElementTree as ET

from chipwhisperer.common.utils import util
import numpy as np

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationError
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunication
from scaaml.capture.scope.lecroy.lecroy_communication import LeCroyCommunicationVisa
from scaaml.capture.scope.lecroy.types import LECROY_CAPTURE_AREA, LECROY_CHANNEL_NAME_T
from scaaml.capture.scope.scope_template import ScopeTemplate
from scaaml.io import Dataset


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

    def post_init(self, dataset: Dataset) -> None:
        """After initialization actions. Saves the scope identity information
        into the capture_info.
        """
        assert self._scope

        # Update capture info with oscilloscope details.
        dataset.capture_info.update(self._scope.get_identity_info())
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
        source_file_path: str = self._scope.call_print_screen(
            capture_area=capture_area)
        self._scope.retrieve_file(source_file_path=source_file_path,
                                  destination_file_path=file_path)
        self._scope.delete_file(file_path=source_file_path)


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
        # Use full precision of measurements
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

    def retrieve_file(self, source_file_path: str,
                      destination_file_path: Path) -> int:
        r"""Transfer a file and return how many bytes were transfered.

        Args:
          source_file_path (str): The file path path on the oscilloscope, e.g.,
            'D:\PRINT-SCREEN--00001.PNG'.
          destination_file_path (Path): The path to the file being saved. Parent
            directories must exist.

        Returns: the number of transferred bytes.
        """
        # Query to get file data from the scope.
        # Answer format:
        # Beware that the #9 (the nine is base 16).
        # TRFL #9nnnnnnnnn<data><crc>
        # Where the TRFL is omitted due to setting "COMM_HEADER OFF"
        # nnnnnnnnn is the file size (bytes in ASCII for decimal
        #   representation) in bytes
        # crc is the 32-bit CRC plus 8-byte CRC trailer
        ans = self._scope_communication.query_binary_values(
            f"TRANSFER_FILE? DISK,HDD,FILE,'{source_file_path}'")

        # With socket we need to strip the header (if there is no header we are
        # using pyvisa).
        strip_header = True
        # Check for start: either "TRFL #9" or "#9"
        if ans[:2] == b"#9":
            # Just number of bytes and CRC
            ans = ans[2:]
        elif ans[:7] == b"TRFL #9":
            # With hader, number of bytes, and CRC
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
        # Setup where to save the file
        hardcopy_setup = (
            f"HARDCOPY_SETUP DEV,PNG,FORMAT,LANDSCAPE,BCKG,BLACK,DEST,FILE,DIR,"
            f"\"D:\",AREA,{capture_area},FILE,\"PRINT-SCREEN.PNG\"")
        self._scope_communication.write(hardcopy_setup)
        hardcopy_setup = self._scope_communication.query("HARDCOPY_SETUP?")
        self._scope_communication.write("SCREEN_DUMP")

        # Parse the full DOS path to the print screen file
        hardcopy_setup = hardcopy_setup.split(",")[8:]
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
        # Delete the file
        self._scope_communication.write(
            f"DELETE_FILE DISK,HDD,FILE,'{file_path}'")
