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
#
# Based on by code kindly shared by Victor Lomné - NinjaLab
"""Communication with the LeCroy oscilloscope. Either using pyvisa or socket.
"""

from abc import ABC, abstractmethod
import hashlib
import logging
import socket
from struct import pack, unpack
from typing import Callable, Optional, ParamSpec, TypeVar, cast
from typing_extensions import Self

import pyvisa
from pyvisa.util import BINARY_DATATYPES

from scaaml.capture.scope.lecroy.lecroy_waveform import LecroyWaveform
from scaaml.capture.scope.lecroy.types import LECROY_CHANNEL_NAME_T

T = TypeVar("T")
Param = ParamSpec("Param")
RetT = TypeVar("RetT")


class LeCroyCommunication(ABC):
    """A class for communication with the LeCroy oscilloscope.
    """

    def __init__(self, ip_address: str, timeout: float = 5.0):
        """Connect to the LeCroy oscilloscope.

        Args:
          ip_address (str): IP address (or hostname) of the oscilloscope.
          timeout (float): Number of seconds before communication timeout.
        """
        self._ip_address = ip_address
        self._timeout = timeout

        self._logger = logging.getLogger("scaaml.capture.scope.lecroy."
                                         "lecroy_communication")

    @abstractmethod
    def connect(self) -> Self:
        """Connect to the scope."""
        return self

    @abstractmethod
    def close(self) -> None:
        """Close the scope connection"""

    @abstractmethod
    def write(self, message: str) -> None:
        """Write a message to the oscilloscope.
        """

    @abstractmethod
    def query(self, message: str) -> str:
        """Query the oscilloscope.
        """

    @abstractmethod
    def get_waveform(self, channel: LECROY_CHANNEL_NAME_T) -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.
        """

    @abstractmethod
    def query_binary_values(self,
                            message: str,
                            container: type[T],
                            datatype: BINARY_DATATYPES = "B") -> T:
        """Query binary data. Beware that the results from socket version might
        contain headers which are stripped by pyvisa.
        """

    def _check_response_template(self) -> None:
        """ Check if the hash of the waveform template matches the supported
        version. This is a workaround, see
        https://github.com/google/scaaml/issues/130
        """
        template = self.query("TMPL?")
        template_hash = hashlib.sha256(template.encode("utf8")).hexdigest()
        if template_hash != LecroyWaveform.SUPPORTED_PROTOCOL_TEMPLATE_SHA:
            self._logger.error(
                "Template description hash is different the expected value. "
                "Template:\n%s", template)
            raise ValueError("Unsupported waveform template description.")


class LeCroyCommunicationError(Exception):
    """Custom exception to deal with various possible exceptions."""


def make_custom_exception(func: Callable[Param, RetT]) -> Callable[Param, RetT]:
    """Decorator to wrap specific exception into a custom one."""

    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetT:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            raise LeCroyCommunicationError("Communication error") from error

    return wrapper


class LeCroyCommunicationVisa(LeCroyCommunication):
    """Use pyvisa to communicate using the LXI protocol over TCP
    ("Utilities > Utilities Setup > Remote" and choose LXI)."""

    def __init__(self, ip_address: str, timeout: float = 5.0):
        super().__init__(
            ip_address=ip_address,
            timeout=timeout,
        )
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._scope: Optional[pyvisa.resources.MessageBasedResource] = None

    @make_custom_exception
    def connect(self) -> Self:  # pragma: no cover
        # For portability and ease of setup we enforce the pure Python backend
        self._resource_manager = pyvisa.ResourceManager("@py")

        scope_resource = self._resource_manager.open_resource(
            f"TCPIP::{self._ip_address}::INSTR",
            resource_pyclass=pyvisa.resources.MessageBasedResource,
        )
        assert isinstance(scope_resource, pyvisa.resources.MessageBasedResource)
        self._scope = scope_resource

        self._scope.timeout = self._timeout * 1_000  # Convert second to ms
        self._scope.clear()

        # Check if the response template is what LecroyWaveform expects
        self._check_response_template()
        return self

    @make_custom_exception
    def close(self) -> None:  # pragma: no cover
        assert self._scope
        self._scope.before_close()
        self._scope.close()
        assert self._resource_manager
        self._resource_manager.close()

    @make_custom_exception
    def write(self, message: str) -> None:  # pragma: no cover
        """Write a message to the oscilloscope.
        """
        assert self._scope
        self._logger.debug("write(message=\"%s\")", message)
        self._scope.write(message)

    @make_custom_exception
    def query(self, message: str) -> str:  # pragma: no cover
        """Query the oscilloscope (write, read, and decode the answer as a
        string).
        """
        assert self._scope
        self._logger.debug("query(message=\"%s\")", message)
        return self._scope.query(message).strip()

    @make_custom_exception
    def get_waveform(
            self, channel: LECROY_CHANNEL_NAME_T
    ) -> LecroyWaveform:  # pragma: no cover
        """Get a LecroyWaveform object representing a single waveform.
        """
        return self.query_binary_values(f"{channel}:WAVEFORM?",
                                        container=LecroyWaveform,
                                        datatype="B")

    @make_custom_exception
    def query_binary_values(
            self,
            message: str,
            container: type[T],
            datatype: BINARY_DATATYPES = "B") -> T:  # pragma: no cover
        """Query binary data."""
        assert self._scope
        self._logger.debug("query_binary_values(message=\"%s\")", message)
        raw_data = self._scope.query_binary_values(
            message,
            datatype=datatype,
            container=container,
        )
        return cast(T, raw_data)


class LeCroyCommunicationSocket(LeCroyCommunication):
    """Use Python socket to communicate using the TCP/IP (VICP).
    ("Utilities > Utilities Setup > Remote" and choose TCPIP)."""

    def __init__(self, ip_address: str, timeout: float = 5.0):
        super().__init__(
            ip_address=ip_address,
            timeout=timeout,
        )
        # Header format (see section "VICP Headers"):
        #   operation: byte
        #   header_version: byte
        #   sequence_number: byte
        #   spare: byte = 0 (reserved for future)
        #   block_length: long = length of the command (block to be sent)
        self._lecroy_command_header = ">4BL"
        self._socket: Optional[socket.socket] = None

    @make_custom_exception
    def connect(self) -> Self:  # pragma: no cover
        assert self._socket is None

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)

        # establish connection
        self._socket.connect((self._ip_address, 1861))

        # Check if the response template is what LecroyWaveform expects
        self._check_response_template()
        return self

    @make_custom_exception
    def close(self) -> None:  # pragma: no cover
        assert self._socket
        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()
        self._socket = None

    @make_custom_exception
    def write(self, message: str) -> None:  # pragma: no cover
        """Write a message to the oscilloscope.
        """
        assert self._socket
        self._socket.send(self._format_command(message))

    @make_custom_exception
    def query(self, message: str) -> str:
        """Query the oscilloscope (write, read, and decode the answer as a
        string).
        """
        return self.query_binary_values(message, container=bytes).decode()

    @make_custom_exception
    def get_waveform(self, channel: LECROY_CHANNEL_NAME_T) -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.

        Args:
          channel (LECROY_CHANNEL_NAME_T): The name of queried channel.
        """
        raw_data = self.query_binary_values(f"{channel}:WAVEFORM?",
                                            container=bytes)
        assert raw_data[:6] == b"ALL,#9"  # followed by 9 digits for size
        len_raw_data = int(raw_data[6:15])  # length without the header
        raw_data = raw_data[15:]
        if len_raw_data + 1 == len(raw_data):
            raw_data = raw_data[:-1]  # last is linefeed
        assert len(raw_data) == len_raw_data
        return LecroyWaveform(raw_data)

    @make_custom_exception
    def query_binary_values(
            self,
            message: str,
            container: type[T],
            datatype: BINARY_DATATYPES = "B") -> T:  # pragma: no cover
        """Query binary data.

        Args:
          message (str): Query message.
          datatype (str): Ignored.
          container: A bytearray is always used.

        Returns: a bytes representation of the response.
        """
        assert self._socket
        del datatype  # ignored
        assert container == bytes

        self._logger.debug("\"%s\"", message)

        # Send message
        self.write(message)

        # Receive and decode answer
        return cast(T, self._get_raw_response())

    def _format_command(self, command: str) -> bytes:
        """Method formatting leCroy command.

        Args:
          command (str): The command to be formatted for sending over a
            socket.

        Returns: bytes representation to be directly sent over a socket.
        """
        # Compute header for the current command, header:
        #   operation = DATA | EOI
        command_header = pack(self._lecroy_command_header, 129, 1, 1, 0,
                              len(command))

        formatted_command = command_header + command.encode("ascii")
        return formatted_command

    def _get_raw_response(self) -> bytes:  # pragma: no cover
        """Get raw response from the socket.

        Returns: bytes representation of the response.
        """
        assert self._socket
        response = bytearray()

        while True:
            header = bytearray()

            # Loop until we get a full header (8 bytes)
            while len(header) < 8:
                header.extend(self._socket.recv(8 - len(header)))

            # Parse formatted response
            (
                operation,
                header_version,  # unused
                sequence_number,  # unused
                spare,  # unused
                total_bytes) = unpack(self._lecroy_command_header, header)

            # Delete unused values
            del header_version
            del sequence_number
            del spare

            # Buffer for the current portion of data
            buffer = bytearray()

            # Loop until we get all data
            while len(buffer) < total_bytes:
                buffer.extend(
                    self._socket.recv(min(total_bytes - len(buffer), 8_192)))

            # Accumulate final response
            response.extend(buffer)

            # Leave the loop when the EOI bit is set
            if operation & 1:
                break

        return bytes(response)
