# Copyright 2022 Google LLC
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
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from scaaml.io import Dataset
from scaaml.capture.crypto_input import AbstractCryptoInput
from scaaml.capture.crypto_alg import AbstractSCryptoAlgorithm
from scaaml.capture.communication import AbstractSCommunication
from scaaml.capture.control import AbstractSControl
from scaaml.capture.scope import AbstractSScope


class AbstractCaptureRunner(ABC):
    """Abstract class for capturing the dataset."""
    def __init__(self, crypto_algorithms: List[AbstractSCryptoAlgorithm],
                 communication: AbstractSCommunication,
                 control: AbstractSControl, scope: AbstractSScope,
                 dataset: Dataset) -> None:
        """Holds all information needed to capture a dataset (using the method
        capture).

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
    def get_crypto_input(self, kt_element) -> AbstractCryptoInput:
        """Process single element from ResumeKTI and return information for
        the crypto algorithm.

        Args:
          kt_element: Single element received from looping over ResumeKTI
            instance. A pair of np arrays.

        Returns: An instance of input for cryptographic algorithm (for example
          an object holding a key and plaintext).
        """

    @abstractmethod
    def get_attack_points_and_measurement(
            self, crypto_alg: AbstractSCryptoAlgorithm,
            crypto_input: AbstractCryptoInput) -> Tuple[Dict, Dict]:
        """Get attack points and measurement. Repeat capture if necessary.
        Raises if hardware fails.

        Args:
          crypto_alg: The object used to get attack points.
          crypto_input: The input from ResumeKTI (a pair of np arrays).

        Returns: Attack points and physical measurement. These are to be used
          directly by scaaml.io.Dataset.write_example.

        Raises: If capturing failed in an unrecoverable way.
        """

    def _stabilize_capture(self, crypto_alg: AbstractSCryptoAlgorithm):
        """Stabilize the capture by capturing a few traces.

        Args:
          crypto_alg: The object used to get stabilization attack points.
        """
        skti = crypto_alg.get_stabilization_kti()
        crypto_input = self.get_crypto_input(next(skti))
        try:
            _, _ = self.get_attack_points_and_measurement(
                crypto_alg=crypto_alg, crypto_input=crypto_input)
        except (Warning, OSError, AssertionError):
            pass
        # Stabilize the capture
        for _ in range(10):
            _, _ = self.get_attack_points_and_measurement(
                crypto_alg=crypto_alg, crypto_input=crypto_input)

    def capture(self):
        """Start (or resume) and finish the capture."""
        self._stabilize_capture(crypto_alg=self._crypto_algorithms[0])
        for crypto_alg in self._crypto_algorithms:
            self._capture_dataset(crypto_alg=crypto_alg)
        self._dataset.check()

    def _capture_dataset(self, crypto_alg):
        """Capture the dataset.

        Args:
          crypto_alg: The object used to get attack points.
        """
        # A part is the id of the shard of a single key, part is in [0, 10].
        part_number = 0
        group_number = 0  # a group is 1 full rotation of the bytes (0 - 255)
        if len(crypto_alg.kti) == 0:
            # Prevent calling self._dataset.close_shard when there was no shard
            # created.
            return
        for trace_number, ktt in enumerate(tqdm(crypto_alg.kti)):
            crypto_input = self.get_crypto_input(ktt)
            attack_points, measurement = self.get_attack_points_and_measurement(
                crypto_alg=crypto_alg, crypto_input=crypto_input)
            # Partition the captures into shards.
            # Captures 0, 1 ,..., examples_per_shard-1 form a single shard.
            if trace_number % crypto_alg.examples_per_shard == 0:
                if trace_number % (crypto_alg.plaintexts *
                                   crypto_alg.repetitions) == 0:
                    # One key has been processed, reset the part_number.
                    part_number = 0
                # Initialize a new shard.
                self._dataset.new_shard(
                    key=crypto_input.key_for_new_shard(),
                    part=part_number,
                    group=group_number,
                    split=crypto_alg.purpose,
                    chip_id=self._control.chip_id,
                )
                part_number += 1
                remaining_traces_in_prev_group = trace_number % (
                    256 * crypto_alg.plaintexts * crypto_alg.repetitions)
                if (trace_number > 0 and remaining_traces_in_prev_group == 0):
                    group_number += 1
            self._dataset.write_example(attack_points=attack_points,
                                        measurement=measurement)
        # Make sure we close the last shard
        self._dataset.close_shard()
