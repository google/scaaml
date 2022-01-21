"""CaptureRunner runs the capture."""
from typing import List
from tqdm.auto import tqdm
import chipwhisperer as cw

from scaaml.io import Dataset
from scaaml.capture.aes.crypto_alg import SCryptoAlgorithm
from scaaml.capture.aes.communication import SCommunication
from scaaml.capture.aes.control import SControl
from scaaml.capture.aes.scope import SScope


class CaptureRunner:
    """Class for capturing the dataset."""
    def __init__(self, crypto_algorithms: List[SCryptoAlgorithm],
                 communication: SCommunication, control: SControl,
                 scope: SScope, dataset: Dataset) -> None:
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
          # Contin capturing the dataset.
          capture_runner.capture()
        """
        self._crypto_algorithms = crypto_algorithms
        self._communication = communication
        self._control = control
        self._scope = scope
        self._dataset = dataset

    def capture_trace(self, key: bytearray, plaintext: bytearray):
        """Try to capture a single trace.

        Args:
          key: The key to use.
          plaintext: The plaintext to be encrypted.

        Returns: The captured trace. None if the capture failed.
          See cw documentation for description of the trace.

        Raises: Warning, OSError
        """
        # Try to capture trace.
        return cw.capture_trace(scope=self._scope.scope,
                                target=self._communication.target,
                                plaintext=plaintext,
                                key=key)

    def get_attack_points_and_measurement(self, key: bytearray,
                                          plaintext: bytearray,
                                          crypto_alg: SCryptoAlgorithm):
        """Get attack points and measurement. Repeat capture if necessary.
        Raises if hardware fails.

        Args:
          key: The key to use.
          plaintext: The plaintext to be encrypted.
          crypto_alg: The object used to get attack points.

        Returns: Attack points and physical measurement.  See cw documentation
          for description of the trace.

        Raises:
          Warning
          OSError
          AssertionError When the textin of the cw trace is different from the
            plaintext.
        """
        while True:  # Make sure to capture the trace.
            trace = self.capture_trace(plaintext=plaintext, key=key)
            if trace:
                assert trace.textin == plaintext
                attack_points = crypto_alg.attack_points(plaintext=plaintext,
                                                         key=key)
                measurement = {
                    "trace": trace.wave,
                }
                return attack_points, measurement

    def _stabilize_capture(self, crypto_alg: SCryptoAlgorithm):
        """Stabilize the capture by capturing a few traces.

        Args:
          crypto_alg: The object used to get stabilization attack points.
        """
        skti = crypto_alg.get_stabilization_kti()
        key, text = next(skti)
        cur_pt = bytearray(text)
        cur_key = bytearray(key)
        try:
            self.capture_trace(plaintext=cur_pt, key=cur_key)
        except (Warning, OSError):
            pass
        # Stabilize the capture
        for _ in range(10):
            self.capture_trace(plaintext=cur_pt, key=cur_key)

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
        for trace_number, (key, text) in enumerate(tqdm(crypto_alg.kti)):
            cur_key = bytearray(key)
            cur_pt = bytearray(text)
            attack_points, measurement = self.get_attack_points_and_measurement(
                key=cur_key, plaintext=cur_pt, crypto_alg=crypto_alg)
            # Partition the captures into shards.
            # Captures 0, 1 ,..., examples_per_shard-1 form a single shard.
            if trace_number % crypto_alg.examples_per_shard == 0:
                if trace_number % (crypto_alg.plaintexts *
                                   crypto_alg.repetitions) == 0:
                    # One key has been processed, reset the part_number.
                    part_number = 0
                # Initialize a new shard.
                self._dataset.new_shard(
                    key=cur_key,
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
