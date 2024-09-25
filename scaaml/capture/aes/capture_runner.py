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
"""CaptureRunner runs the capture."""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

import chipwhisperer as cw
from chipwhisperer.common.traces import Trace
from chipwhisperer.common.utils import util as cw_util

from scaaml.capture.capture_runner import AbstractCaptureRunner
from scaaml.capture.crypto_input import AbstractCryptoInput
from scaaml.capture.aes.crypto_input import CryptoInput
from scaaml.capture.crypto_alg import AbstractSCryptoAlgorithm


class CaptureRunner(AbstractCaptureRunner):
    """Class for capturing the dataset."""

    def get_crypto_input(self, kt_element: NamedTuple) -> AbstractCryptoInput:
        """Process single element from ResumeKTI and return information for
        the crypto algorithm.

        Args:
          kt_element: Single element received from looping over ResumeKTI
            instance. A dictionary of np arrays.

        Returns: An instance of input of cryptographic algorithm (an object
          holding a key and plaintext).
        """
        return CryptoInput(kt_element=kt_element)

    def capture_trace(self, crypto_input: CryptoInput) -> Optional[Trace]:
        """Try to capture a single trace.

        Args:
          crypto_input: The input for the cryptographic algorithm.

        Returns: The captured trace. None if the capture failed.
          See cw documentation for description of the trace.

        Raises:
          Warning, OSError: If cw.capture_trace raises.
          AssertionError: If the textin in the trace is different from
            plaintext.
        """
        # Convert to cw bytearray, which has nicer __str__ and __repr__.
        plaintext = cw_util.bytearray(crypto_input.plaintext)
        key = cw_util.bytearray(crypto_input.key)

        # Get the scope object.
        scope = self._scope.scope
        assert scope is not None

        # Get a target from Optional[TargetTypes].
        target = self._communication.target
        assert target is not None

        # Capture the trace.
        # TODO(issue #79): Allow typechecking of this call.
        trace = cw.capture_trace(
            scope=scope,  # type: ignore[arg-type]
            target=target,
            plaintext=plaintext,
            key=key)

        return trace

    def get_attack_points_and_measurement(
        self, crypto_alg: AbstractSCryptoAlgorithm,
        crypto_input: AbstractCryptoInput
    ) -> Tuple[Dict[str, bytearray], Dict[str, npt.NDArray[np.generic]]]:
        """Get attack points and measurement. Repeat capture if necessary.
        Raises if hardware fails.

        Args:
          crypto_alg (AbstractSCryptoAlgorithm): The object used to get
            attack points.
          crypto_input (CryptoInput): Representation of the input of the
            cryptographic algorithm.

        Returns: Attack points and physical measurement. These are to be used
          directly by scaaml.io.Dataset.write_example. Returns a pair of
          dictionaries (attack_points, measurement).

        Raises: If capturing failed in an unrecoverable way.
        """
        assert isinstance(crypto_input, CryptoInput)
        while True:  # Make sure to capture the trace.
            trace = self.capture_trace(crypto_input=crypto_input)
            if trace:
                assert trace.textin == crypto_input.plaintext
                attack_points = crypto_alg.attack_points(
                    plaintext=crypto_input.plaintext,
                    key=crypto_input.key,
                )
                assert trace.textout == attack_points["ciphertext"]
                measurement = {
                    "trace1": trace.wave,
                }
                return attack_points, measurement
