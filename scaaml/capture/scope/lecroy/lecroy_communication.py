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
import hashlib
import logging
from typing import Optional

import pyvisa

from scaaml.capture.scope.lecroy.lecroy_waveform import LecroyWaveform
from scaaml.capture.scope.lecroy.types import LECROY_CHANNEL_NAME_T


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
    def get_waveform(self, channel: LECROY_CHANNEL_NAME_T) -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.
        """

    @abstractmethod
    def query_binary_values(self,
                            message: str,
                            datatype="B",
                            container=bytearray) -> bytearray:
        """Query binary data."""

    def _check_response_template(self):
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


def make_custom_exception(func):
    """Decorator to wrap specific exception into a custom one."""

    def wrapper(*args, **kwargs):
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
    def connect(self):
        # For portability and ease of setup we enforce the pure Python backend
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

        # Check if the response template is what LecroyWaveform expects
        self._check_response_template()

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
        self._logger.debug("write(message=\"%s\")", message)
        self._scope.write(message)

    @make_custom_exception
    def query(self, message: str) -> str:
        """Query the oscilloscope (write, read, and decode the answer as a
        string).
        """
        assert self._scope is not None
        self._logger.debug("query(message=\"%s\")", message)
        return self._scope.query(message).strip()

    @make_custom_exception
    def get_waveform(self, channel: LECROY_CHANNEL_NAME_T) -> LecroyWaveform:
        """Get a LecroyWaveform object representing a single waveform.
        """
        assert self._scope is not None

        return self._scope.query_binary_values(
            f"{channel}:WAVEFORM?",
            datatype="B",
            container=LecroyWaveform,
        )  # type: ignore

    @make_custom_exception
    def query_binary_values(self,
                            message: str,
                            datatype="B",
                            container=bytearray):
        """Query binary data."""
        assert self._scope is not None
        self._logger.debug("query_binary_values(message=\"%s\")", message)
        return self._scope.query_binary_values(
            message,
            datatype=datatype,
            container=container,
        )
