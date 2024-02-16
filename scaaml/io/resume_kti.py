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
"""Resumable iterator for key-text pairs (or more general tuples) with
autosaves.

Autosaves after shard length iterations.

Typical usage example:

  # Save all encryption parameter tuples ahead before the experiment starts.
  # If kt_filename exists this does nothing.
  create_resume_kti(parameters={"keys": keys, "texts": texts},
                    shard_length=shard_length,
                    kt_filename="parameters_tuples.txt",
                    progress_filename="progress_tuples.txt")

  # Load the savepoint (all the tuples + the index where to continue) at the
  # start or resume of the experiment.
  resume_kti = ResumeKTI(kt_filename="parameters_tuples.txt",
                         progress_filename="progress_tuples.txt"

  # Loop with a progress-bar (progress bar has always the same length, but when
  # restarted the progress is not lost).
  # for key, text in tqdm(resume_kti, initial=resume_kti.initial_index):
  for current_parameters in resume_kti:
      # Note that the names in the dictionary are the same as in the parameters.
      # Thus if you use "keys" for an array of keys then you get plural even for
      # the single iterated key.
      # current_parameters = { "keys": keys[i], "texts": texts[i] }
      # Make sure to capture the trace.
      while True:
          # Try to capture trace.
          if trace:
              break

File format:

  Parameter tuples file is a binary file with numpy scalar and arrays. The
  contents are shard length, array of strings with parameter names and then
  arrays of parameters in the same order. The array of parameter values are
  interpreted in the way that (parameter1[i], parameter2[i], ...) is a single
  key-text tuple for any i in range(len(parameter1)), thus all have the same
  length. This is done in order to allow different data type of key and text
  (for instance AES256 where key is 256 bits and text is 128 bits) and different
  number of saved parameters (saving masks...).

  Progress file is a text file with a single integer -- the number of traces
  captured so far. The number in it is a multiple of the shard length. This file
  is automatically updated after shard length iterations over ResumeKTI.
"""

from collections import namedtuple
from typing import Any, Dict, List
from typing_extensions import Self

import numpy as np
import numpy.typing as npt


def get_any_value_len(parameters: Dict[str, npt.NDArray[np.generic]]) -> int:
    """Convenience function that returns the len of any value. All lengths
    should be the same (so that we can iterate).

    Args:
      parameters (Dict[str, np.ndarray]): Parameters of the encryption (keys,
        plaintexts, masks...). A dictionary with name and the corresponding
        parameters represented as an np.array. All values must have the same
        non-zero length, their length must be a multiple of shard_length.

    Returns: Length of a single value.
    """
    return len(next(iter(parameters.values())))


def check_lengths(parameters: Dict[str, npt.NDArray[np.generic]],
                  shard_length: np.uint64) -> None:
    """Check that all parameters have the same non-zero length.

    Args:
      parameters (Dict[str, np.ndarray]): Parameters of the encryption (keys,
        plaintexts, masks...). A dictionary with name and the corresponding
        parameters represented as an np.array. All values must have the same
        non-zero length, their length must be a multiple of shard_length.
      shard_length (np.uint64): How many tuples are in a shard. ResumeKTI
        autosaves progress after finishing a shard.

    Raises: ValueError if some length is not right.
    """
    if not parameters:
        raise ValueError("No parameters.")

    # Get length of a single values array.
    length = get_any_value_len(parameters)
    for parameter, values in parameters.items():
        if len(values) == 0:
            raise ValueError(f"There are no parameters for {parameter}")
        if len(values) % int(shard_length) != 0:
            raise ValueError(f"The number of values of {parameter} is not "
                             f"divisible by shard_length.")
        if len(values) != length:
            raise ValueError("There are different number of parameter values.")


def _create_resume_kti(parameters: Dict[str, npt.NDArray[np.generic]],
                       shard_length: np.uint64,
                       kt_filename: str = "parameters_tuples.txt",
                       progress_filename: str = "progress_tuples.txt",
                       allow_pickle: bool = False) -> None:
    """Saves the parameter tuples for later use. See create_resume_kti
    convenience function.

    Args:
        parameters (Dict[str, np.ndarray]): Parameters of the encryption (keys,
          plaintexts, masks...). A dictionary with name and the corresponding
          parameters represented as an np.array. All values must have the same
          non-zero length, their length must be a multiple of shard_length.
        shard_length (np.uint64): How many tuples are in a shard. ResumeKTI
          autosaves progress after finishing a shard.
        kt_filename: File where to save shard_length and parameters. If the
          file exists prints a message, but does not overwrite.
        progress_filename: The file with the number of traces captured so far.
        allow_pickle: Parameter for numpy.save.
    """
    # Check parameter lengths.
    check_lengths(parameters=parameters, shard_length=shard_length)

    # Save into file.
    try:
        with open(kt_filename, "xb") as kt_file:
            # Save the shard length.
            np.save(kt_file, shard_length, allow_pickle=allow_pickle)

            # Saving a dictionary directly would require allow_pickle=True. Save
            # dictionary.keys as an array of strings and then save the values in
            # the same order as the dictionary.keys array.
            parameter_names: List[str] = list(parameters.keys())
            np.save(kt_file,
                    np.array(parameter_names),
                    allow_pickle=allow_pickle)

            # Save their values.
            for parameter_name in parameter_names:
                np.save(kt_file,
                        parameters[parameter_name],
                        allow_pickle=allow_pickle)
    except FileExistsError:
        print(f"File {kt_filename} already exists, if you want to generate new"
              " keys, remove it first.")
        return
    with open(progress_filename, "w", encoding="utf-8") as progress_file:
        progress_file.write("0")


class ResumeKTI:
    """Iterable object, auto-saves progress after iterating shard length times.

    Does not support multiple iterations. If this is needed, write zero to
    progress_filename and create a new ResumeKTI object.
    """

    def __init__(self,
                 kt_filename: str = "parameters_tuples.txt",
                 progress_filename: str = "progress_tuples.txt",
                 allow_pickle: bool = False) -> None:
        """Create a resumable key-text iterable.

        Args:
            kt_filename: File to load shard_length and parameters.
            progress_filename: The file with the number of traces captured so
              far.  Gets updated by iterating shard length times or manually
              calling _save_progress.
            allow_pickle: Parameter for numpy.load.
        """
        with open(kt_filename, "rb") as kt_tuples:
            # Load shard length.
            self._shard_length = np.load(kt_tuples, allow_pickle=allow_pickle)

            # Load list of parameter names.
            parameter_names = np.load(kt_tuples, allow_pickle=allow_pickle)
            # Create the namedtuple class. Ignore mypy warning that named
            # tuples should not be constructed dynamically and that a list or
            # tuple literal should be used for the names.
            self._element_class = namedtuple(  # type: ignore[misc]
                "EncryptionParameters", parameter_names)

            # Load all parameters.
            self._parameters = {}
            for parameter_name in parameter_names:
                self._parameters[parameter_name] = np.load(
                    kt_tuples, allow_pickle=allow_pickle)

        # Check length of parameters.
        check_lengths(parameters=self._parameters,
                      shard_length=self._shard_length)

        # Load previous progress.
        self._progress_filename = progress_filename
        with open(self._progress_filename, encoding="utf-8") as progress_file:
            # Ensure self._index is a multiple of self._shard_length.
            i: int = int(progress_file.read())
            self._index = int(i - (i % self._shard_length))
        # Check validity of the progress.
        assert 0 <= self._index <= get_any_value_len(self._parameters)
        # How many traces have been captured before creating this object.
        self._initial_index = self._index

    def _save_progress(self) -> None:
        """Save the current iteration index.

        Save the number of traces in all finished shards (how many finished
        traces have been captured inside finished shards).  Updates content of
        the file self._progress_filename.

        Not atomic. Calling this 1000 times takes ~1s.
        """
        with open(self._progress_filename, "w",
                  encoding="utf-8") as progress_file:
            # Ensure the saved index is the largest multiple of
            # self._shard_length which is at most self._index.
            i = self._index - (self._index % self._shard_length
                              )  # End of last shard.
            i = int(i)  # Modulo by np.uint64 produces np.float64
            progress_file.write(str(i))

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        """Next with auto-save.

        Auto-save when a new shard is starting (after each shard length + 1
        calls without a _save_progress call), thus updates content of
        self._progress_filename file.

        Returns: Encryption parameters as a namedtuple.
        """
        if self._index % self._shard_length == 0:
            self._save_progress()
        if self._index >= get_any_value_len(self._parameters):
            raise StopIteration
        self._index += 1

        # Construct the namedtuple to return.
        element_class_parameters = {
            parameter_name: parameter_values[self._index - 1]
            for parameter_name, parameter_values in self._parameters.items()
        }
        return self._element_class(**element_class_parameters)

    def __len__(self) -> int:
        """How many tuples are there in total (including the initial_index
        skipped tuples)."""
        return get_any_value_len(self._parameters)

    @property
    def initial_index(self) -> int:
        """Returns where the current capture has started. The result is
        divisible by shard_length (number of examples in a shard) unless the
        progress file has been corrupted.

        Returns: The number of examples captured before starting this
        iteration.
        """
        return self._initial_index


def create_resume_kti(parameters: Dict[str, npt.NDArray[np.generic]],
                      shard_length: np.uint64,
                      kt_filename: str = "parameters_tuples.txt",
                      progress_filename: str = "progress_tuples.txt",
                      allow_pickle: bool = False) -> ResumeKTI:
    """Saves the parameter tuples and returns a ResumeKTI object for immediate
    use.

    Args:
        parameters (Dict[str, np.ndarray]): Parameters of the encryption (keys,
          plaintexts, masks...). A dictionary with name and the corresponding
          parameters represented as an np.array. All values must have the same
          non-zero length, their length must be a multiple of shard_length.
        shard_length (np.uint64): How many tuples are in a shard. ResumeKTI
          autosaves progress after finishing a shard.
        kt_filename: File where to save shard_length and parameters. If the
          file exists prints a message, but does not overwrite.
        progress_filename: The file with the number of traces captured so far.
        allow_pickle: Parameter for numpy.save.

    Returns: ResumeKTI object just created.
    """
    _create_resume_kti(parameters=parameters,
                       shard_length=shard_length,
                       kt_filename=kt_filename,
                       progress_filename=progress_filename,
                       allow_pickle=allow_pickle)
    return ResumeKTI(kt_filename=kt_filename,
                     progress_filename=progress_filename,
                     allow_pickle=allow_pickle)
