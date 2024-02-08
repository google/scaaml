# Copyright 2022-2024 Google LLC
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
"""The class that represents input of the AES cryptographic algorithm."""

from typing import Any, Dict

from scaaml.capture.crypto_input import AbstractCryptoInput


class CryptoInput(AbstractCryptoInput):
    """Single instance of cryptographic input for AES."""

    def __init__(self, kt_element: Any) -> None:
        """Initialize the crypto input.

        Args:
          kt_element (namedtuple): An element returned by iteration over
            ResumeKTI (namedtuple of np.arrays with key, plaintext).

        Example use:
           from scaaml.aes_forward import AES

           crypto_input = CryptoInput(kt_element)
           ap_name = "sub_bytes_in"
           sub_bytes_in = AES.get_attack_point(name=ap_name,
                                               **crypto_input.kwargs())
        """
        super().__init__()
        self._key = bytearray(kt_element.keys)
        self._plaintext = bytearray(kt_element.texts)

    def key_for_new_shard(self) -> bytearray:
        """Return the key parameter of scaaml.io.Dataset.new_shard."""
        return self._key

    def kwargs(self) -> Dict[str, bytearray]:
        """Return keyword arguments for getting an attack point.

        Example use:
           from scaaml.aes_forward import AES

           crypto_input = CryptoInput(kt_element)
           ap_name = "sub_bytes_in"
           sub_bytes_in = AES.get_attack_point(name=ap_name,
                                               **crypto_input.kwargs())
        """
        return {
            "key": self._key,
            "plaintext": self._plaintext,
        }

    def __str__(self) -> str:
        """String representation for debugging purposes."""
        return f"key: {self._key} plaintext: {self._plaintext}"

    @property
    def key(self) -> bytearray:
        """Return the key."""
        return self._key

    @property
    def plaintext(self) -> bytearray:
        """Return the plaintext."""
        return self._plaintext
