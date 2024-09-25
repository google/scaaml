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
"""CaptureRunner runs the capture."""
from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Sequence, Tuple
from tqdm.auto import tqdm

import numpy as np
import numpy.typing as npt

from scaaml.io import Dataset
from scaaml.io import DatasetFiller
from scaaml.capture.crypto_input import AbstractCryptoInput
from scaaml.capture.crypto_alg import AbstractSCryptoAlgorithm
from scaaml.capture.communication import AbstractSCommunication
from scaaml.capture.control import AbstractSControl
from scaaml.capture.scope import AbstractSScope


class AbstractCaptureRunner(ABC):
    """Abstract class for capturing the dataset."""

    def __init__(self, *, crypto_algorithms: Sequence[AbstractSCryptoAlgorithm],
                 communication: AbstractSCommunication,
                 control: AbstractSControl, scope: AbstractSScope,
                 dataset: Dataset) -> None:
        """Holds all information needed to capture a dataset (using the method
        `capture`).

        Args:
          crypto_algorithms: Provide attack points and information about the
            encryption algorithm. A separate capture is run for each element.
          communication: Object that communicates with the observed chip.
          control: Control of the target board.
          scope: Scope which does the measurements.
          dataset: Dataset to save measurements into.

        Typical usage example:
          capture_runner = CaptureRunner(crypto_algorithms=[crypto_alg],
                                         scope=scope,
                                         communication=target,
                                         control=control,
                                         dataset=dataset)
          # Continue capturing the dataset.
          capture_runner.capture()
        """
        self._crypto_algorithms = crypto_algorithms
        self._communication = communication
        self._control = control
        self._scope = scope
        self._dataset = dataset

    @abstractmethod
    def get_crypto_input(self, kt_element: NamedTuple) -> AbstractCryptoInput:
        """Process single element from ResumeKTI and return information for
        the crypto algorithm.

        Args:
          kt_element (namedtuple): Single element received from looping over
            ResumeKTI instance. A namedtuple of np arrays.

        Returns: An instance of input for cryptographic algorithm (for example
          an object holding a key and plaintext).
        """

    @abstractmethod
    def get_attack_points_and_measurement(
        self, crypto_alg: AbstractSCryptoAlgorithm,
        crypto_input: AbstractCryptoInput
    ) -> Tuple[Dict[str, bytearray], Dict[str, npt.NDArray[np.generic]]]:
        """Get attack points and measurement. Repeat capture if necessary.
        Raises if hardware fails.

        Args:
          crypto_alg (AbstractSCryptoAlgorithm): The object used to get attack
            points.
          crypto_input (AbstractCryptoInput): The input for encryption.

        Returns: Attack points and physical measurement. These are to be used
          directly by scaaml.io.Dataset.write_example.

        Raises: If capturing failed in an unrecoverable way.
        """

    def _stabilize_capture(self, crypto_alg: AbstractSCryptoAlgorithm) -> None:
        """Stabilize the capture by capturing a few traces.

        Args:
          crypto_alg: The object used to get stabilization attack points.
        """
        crypto_input = self.get_crypto_input(next(crypto_alg.stabilization_kti))
        try:
            _, _ = self.get_attack_points_and_measurement(
                crypto_alg=crypto_alg, crypto_input=crypto_input)
        except (Warning, OSError, AssertionError):
            pass
        # Stabilize the capture
        for _ in range(10):
            _, _ = self.get_attack_points_and_measurement(
                crypto_alg=crypto_alg, crypto_input=crypto_input)

        # Save an image of a trace
        # TODO(#183) Plot the measured trace
        if hasattr(self._scope, "print_screen"):
            # Use native print_screen
            print_screen_fn = getattr(self._scope, "print_screen")
            assert callable(print_screen_fn)
            print_screen_fn(self._dataset.path / "print_screen.png")

    def capture(self) -> None:
        """Start (or resume) and finish the capture."""
        self._stabilize_capture(crypto_alg=self._crypto_algorithms[0])
        for crypto_alg in self._crypto_algorithms:
            self._capture_dataset(crypto_alg=crypto_alg)
        self._dataset.check()

    def _capture_dataset(self, crypto_alg: AbstractSCryptoAlgorithm) -> None:
        """Capture the dataset.

        Args:
          crypto_alg: The object used to get attack points.
        """
        # Save oscilloscope settings (e.g., sampling rate).
        # TODO ideally this should be a representation which is easy to
        # parse and use to create a new scope instance. However we need to
        # support ChipWhisperer scopes and our custom scopes at the same
        # time.
        self._dataset.capture_info[f"scope_{crypto_alg.purpose}"] = repr(
            self._scope.scope)

        assert crypto_alg.kti is not None
        # TODO: crypto_alg.kti is generically typed as Iterator[Any] but would
        # need a proper base class instead that implements the `initial_index`
        # property.
        # In practice it can be a `ResumeKTI` which has it or a
        # `AcqKeyTextPatternScaaml` which doesn't.
        skip_examples: int = getattr(crypto_alg.kti, "initial_index", 0)
        # Context manager properly opens and closes shards.
        with DatasetFiller(
                dataset=self._dataset,
                # The values plaintexts_per_key and repetitions are used to
                # determine the group_number which is never used.
                plaintexts_per_key=1,
                repetitions=1,
                skip_examples=skip_examples,
        ) as dataset_filler:
            # Add examples, new shards are opened automatically.
            for ktt in tqdm(crypto_alg.kti, initial=skip_examples):
                crypto_input = self.get_crypto_input(ktt)
                (
                    attack_points,
                    measurement,
                ) = self.get_attack_points_and_measurement(
                    crypto_alg=crypto_alg,
                    crypto_input=crypto_input,
                )
                dataset_filler.write_example(
                    attack_points=attack_points,
                    measurement=measurement,
                    current_key=list(crypto_input.key_for_new_shard()),
                    split_name=crypto_alg.purpose,
                    chip_id=self._control.chip_id,
                )
