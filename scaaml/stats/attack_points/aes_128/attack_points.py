# Copyright 2025 Google LLC
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
"""Attack points.
"""
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

from scaaml.stats.attack_points.aes_128.full_aes import key_schedule, SBOX, SBOX_INV
from scaaml.aes_forward import AESSBOX


class AttackPointAES128(ABC):

    @classmethod
    def all_subclasses(cls) -> set[type]:
        """Recursively get all subclasses of the class."""
        subclasses: set[type] = set(cls.__subclasses__())
        for subclass in cls.__subclasses__():
            subclasses.update(subclass.all_subclasses())
        return subclasses

    @classmethod
    def subclass_names(cls) -> list[str]:
        """Return a list of all subclass names (as strings)."""
        return [subclass.__name__ for subclass in cls.all_subclasses()]

    @classmethod
    def from_name(cls, name: str) -> type["AttackPointAES128"]:
        """Return the subclass of AttackPointAES128 with the given name."""
        for subclass in cls.all_subclasses():
            if subclass.__name__ == name:
                return subclass
        raise ValueError(f"No subclass {cls.__name__} with the name"
                         f"'{name}' found.")

    @classmethod
    def leakage_knowing_secrets(cls, key: npt.NDArray[np.uint8],
                                plaintext: npt.NDArray[np.uint8],
                                byte_index: int) -> int:
        """When we know all the information.
        """
        guess = cls.target_secret(
            key=key,
            plaintext=plaintext,
            byte_index=byte_index,
        )
        ciphertext = np.array(
            AESSBOX.ciphertext(
                key=bytearray(np.array(key, dtype=np.uint8)),
                plaintext=bytearray(np.array(plaintext, dtype=np.uint8)),
            ),
            dtype=np.uint8,
        )
        return cls.leakage_from_guess(
            plaintext=plaintext,
            ciphertext=ciphertext,
            guess=guess,
            byte_index=byte_index,
        )

    @classmethod
    @abstractmethod
    def leakage_from_guess(cls, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8], guess: int,
                           byte_index: int) -> int:
        """We know only public information and a guess of the hidden value.
        """

    @classmethod
    @abstractmethod
    def target_secret(cls, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8], byte_index: int) -> int:
        """When we know all the information. Return the hidden value we are
        trying to guess. For instance we could be guessing the last key
        schedule round of AES.

        The following holds for all subclasses:
        ```
        assert issubclass(cls, AttackPointAES128)
        byte_index = np.random.randint(0, 16)
        key = np.random.randint(0, 256, 16).astype(np.uint8)
        plaintext = np.random.randint(0, 256, 16).astype(np.uint8)
        ciphertext = AESSBOX.ciphertext(key=key, plaintext=plaintext)

        guess: int = cls.target_secret(
            key=key,
            plaintext=plaintext,
            byte_index=byte_index,
        )
        a: int = cls.leakage_knowing_secrets(
            key=key,
            plaintext=plaintext,
            byte_index=byte_index,
        )
        b: int = cls.leakage_from_guess(
            plaintext=plaintext,
            ciphertext=ciphertext,
            guess=guess,
            byte_index=byte_index,
        )
        assert a == b
        ```
        """

    @staticmethod
    def different_target_secrets() -> int:
        """How many different secret possibilities there are to guess. Since we
        are targetting a byte value the result is 256.
        """
        return 256


class Plaintext(AttackPointAES128):
    """The plaintext value is leaking.
    """

    @classmethod
    def leakage_from_guess(cls, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8], guess: int,
                           byte_index: int) -> int:
        assert 0 <= guess < 256
        return int(guess)

    @classmethod
    def target_secret(cls, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8], byte_index: int) -> int:
        return int(plaintext[byte_index])


class SubBytesIn(AttackPointAES128):
    """Input of the first S-BOX is leaking.
    """

    @classmethod
    def leakage_from_guess(cls, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8], guess: int,
                           byte_index: int) -> int:
        assert 0 <= guess < 256
        return int(guess ^ plaintext[byte_index])

    @classmethod
    def target_secret(cls, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8], byte_index: int) -> int:
        return int(key[byte_index])


class SubBytesOut(AttackPointAES128):
    """Output of the first S-BOX is leaking.
    """

    @classmethod
    def leakage_from_guess(cls, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8], guess: int,
                           byte_index: int) -> int:
        assert 0 <= guess < 256
        return int(SBOX[guess ^ plaintext[byte_index]])

    @classmethod
    def target_secret(cls, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8], byte_index: int) -> int:
        return int(key[byte_index])


class LastRoundStateDiff(AttackPointAES128):
    """Difference of last round states is leaking. Useful for instance for
    hardware AES (see `https://github.com/newaetech/chipwhisperer-jupyter`
    `courses/sca201/Lab 2_2 - CPA on Hardware AES Implementation.ipynb`)
    """

    @classmethod
    def leakage_from_guess(cls, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8], guess: int,
                           byte_index: int) -> int:
        assert 0 <= guess < 256

        invshift_undo = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

        st10 = ciphertext[invshift_undo[byte_index]]
        st9 = SBOX_INV[ciphertext[byte_index] ^ guess]
        byte_value = st9 ^ st10

        return int(byte_value)

    @classmethod
    def target_secret(cls, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8], byte_index: int) -> int:
        last_key_schedule = key_schedule(key)
        correct_k = last_key_schedule[-4:].reshape(-1)
        guess = correct_k[byte_index]

        return int(guess)


class LeakageModelAES128:
    """The value that might be correlated with value(s) in the trace. Keeps the
    byte index and if we should use just the Hamming weight.
    """

    def __init__(self,
                 byte_index: int,
                 attack_point: AttackPointAES128,
                 use_hamming_weight: bool = True) -> None:
        """Gives the leakage function.

        Args:

          byte_index (int): Which byte to target (in range(16)).

          attack_point (AttackPointAES128): Use either input or output of the
          first SBOX.

          use_hamming_weight (bool): Use just the Hamming weight of the value.
        """
        assert 0 <= byte_index < 16
        self._byte_index: int = byte_index
        self._use_hamming_weight: bool = use_hamming_weight
        self._attack_point: AttackPointAES128 = attack_point

    @property
    def different_target_secrets(self) -> int:
        """How many different values does the secret attain.
        """
        return self._attack_point.different_target_secrets()

    @property
    def different_leakage_values(self) -> int:
        """How many possible values of the leakage there are.
        """
        if self._use_hamming_weight:
            return 9
        # Full byte.
        return 256

    @staticmethod
    def _safe_cast(value: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Ideally this function does nothing. But it is very easy to pass an
        array of larger dtype and then be surprised by results of leakage (the
        additional bytes are zero).

        Args:

          value (np.typing.NDArray[np.uint8]): Try to convert this to uint8.

        Raises ValueError if the values cannot be safely converted to uint8.
        """
        uint8_value = np.array(value, dtype=np.uint8)
        if not (uint8_value == value).all():
            raise ValueError("Conversion to uint8 was not successful.")
        return uint8_value

    def leakage_knowing_secrets(self, plaintext: npt.NDArray[np.uint8],
                                key: npt.NDArray[np.uint8]) -> int:
        """Return the leakage value.

        Args:

          plaintext (np.typing.NDArray[np.uint8]): Array of byte values. The
          method fails if there is a value which cannot be converted to uint8.

          key (np.typing.NDArray[np.uint8]): Array of byte values. The method
          fails if there is a value which cannot be converted to uint8.

        Returns: An integer representing the leakage.
        """
        plaintext = self._safe_cast(plaintext)
        key = self._safe_cast(key)

        # Get the byte value of the leakage.
        byte_value: int = self._attack_point.leakage_knowing_secrets(
            key=key,
            plaintext=plaintext,
            byte_index=self._byte_index,
        )

        # Maybe convert to Hamming weight.
        if self._use_hamming_weight:
            return int(byte_value).bit_count()

        return byte_value

    def leakage_from_guess(self, plaintext: npt.NDArray[np.uint8],
                           ciphertext: npt.NDArray[np.uint8],
                           guess: int) -> int:
        """The leakage (possibly its Hamming weight) given a guess of the
        secret.
        """
        plaintext = self._safe_cast(plaintext)
        ciphertext = self._safe_cast(ciphertext)

        # Get the byte value of the leakage.
        byte_value: int = self._attack_point.leakage_from_guess(
            plaintext=plaintext,
            ciphertext=ciphertext,
            guess=guess,
            byte_index=self._byte_index,
        )

        # Maybe convert to Hamming weight.
        if self._use_hamming_weight:
            return int(byte_value).bit_count()

        return byte_value

    def target_secret(self, key: npt.NDArray[np.uint8],
                      plaintext: npt.NDArray[np.uint8]) -> int:
        """The secret we are trying to guess.
        """
        plaintext = self._safe_cast(plaintext)
        key = self._safe_cast(key)

        # Get the byte value of the leakage.
        return self._attack_point.target_secret(
            key=key,
            plaintext=plaintext,
            byte_index=self._byte_index,
        )
