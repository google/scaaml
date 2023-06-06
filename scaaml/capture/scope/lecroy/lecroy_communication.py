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
"""Communication with the LeCroy oscilloscope. Either using pyvisa or socket.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pyvisa

from scaaml.capture.scope.lecroy.lecroy_waveform import LecroyWaveform


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

    @abstractmethod
    def connect(self):
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
    def get_waveform(self, channel: str) -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.
        """

    @abstractmethod
    def query_binary_values(self,
                            message: str,
                            datatype="B",
                            container=None) -> bytearray:
        """Query binary data."""


class LeCroyCommunicationError(Exception):
    """Custom exception to deal with various possible exceptions."""


def make_custom_exception(func):
    """Decorator to wrap specific exception into a custom one."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            raise LeCroyCommunicationError("Communication error") from error

    return wrapper


class LeCroyCommunicationLXI(LeCroyCommunication):
    """Use pyvisa to communicate."""

    def __init__(self, ip_address: str, timeout: float = 5.0):
        super().__init__(
            ip_address=ip_address,
            timeout=timeout,
        )
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._scope: Optional[pyvisa.resources.MessageBasedResource] = None

    @make_custom_exception
    def connect(self):
        self._resource_manager = pyvisa.ResourceManager("@py")
        assert self._resource_manager is not None

        scope_resource = self._resource_manager.open_resource(
            f"TCPIP::{self._ip_address}::INSTR",
            resource_pyclass=pyvisa.resources.MessageBasedResource,
        )
        assert isinstance(scope_resource, pyvisa.resources.MessageBasedResource)
        self._scope = scope_resource

        assert self._scope is not None
        self._scope.timeout = self._timeout * 1_000  # Convert second to ms
        self._scope.clear()

    @make_custom_exception
    def close(self) -> None:
        assert self._scope is not None
        self._scope.before_close()
        self._scope.close()
        assert self._resource_manager is not None
        self._resource_manager.close()

    @make_custom_exception
    def write(self, message: str) -> None:
        """Write a message to the oscilloscope.
        """
        assert self._scope is not None
        self._scope.write(message)

    @make_custom_exception
    def query(self, message: str) -> str:
        """Query the oscilloscope.
        """
        assert self._scope is not None
        return self._scope.query(message).strip()

    @make_custom_exception
    def get_waveform(self, channel: str = "1") -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.
        """
        assert self._scope is not None
        return self._scope.query_binary_values(
            f"C{channel}:WAVEFORM?",
            datatype="B",
            container=LecroyWaveform,
        )  # type: ignore

    @make_custom_exception
    def query_binary_values(self, message: str, datatype="B", container=None):
        """Query binary data."""
        assert self._scope is not None
        return self._scope.query_binary_values(
            message,
            datatype=datatype,
            container=container,
        )
