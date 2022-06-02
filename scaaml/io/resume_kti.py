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
"""Resumeable iterator for key-text pairs with autosaves.

Autosaves after shard length iterations.

Typical usage example:

  # Save all key-text pairs ahead before the experiment starts.
  # If kt_filename exists this does nothing.
  create_resume_kti(keys, texts, shard_length=shard_length,
                    kt_filename="key_text_pairs.txt",
                    progress_filename="progress_pairs.txt")

  # Load the savepoint (all the pairs + the index where to continue) at the
  # start or resume of the experiment.
  resume_kti = ResumeKTI(kt_filename="key_text_pairs.txt",
                         progress_filename="progress_pairs.txt"

  # for key, text in tqdm(resume_kti):  # Loop with a progress-bar.
  for key, text in resume_kti:
      # Make sure to capture the trace.
      while True:
          # Try to capture trace.
          if trace:
              break

File format:

  Text-key pairs file is a binary file with numpy scalar and two arrays.  The
  contents are shard length, keys, texts. The array of keys and texts are
  interpreted in the way that (keys[i], texts[i]) is a single key-text pair for
  any i in range(len(keys)), thus both have the same length. This is done in
  order to allow different data type of key and text (for instance AES256 where
  key is 256 bits and text is 128 bits).

  Progress file is a text file with a single integer -- the number of traces
  captured so far. The number in it is a multiple of the shard length. This file
  is automatically updated after shard length iterations over ResumeKTI.
"""

import numpy as np


def _create_resume_kti(keys: np.ndarray,
                       texts: np.ndarray,
                       shard_length: np.uint64,
                       kt_filename: str = "key_text_pairs.txt",
                       progress_filename: str = "progress_pairs.txt",
                       allow_pickle: bool = False) -> None:
    """Saves the key-text pairs for later use. See create_resume_kti convenience
    function.

    Args:
        keys: Numpy array of keys to capture traces with. Non-zero length,
          length is a multiple of shard_length. Different array for keys and
          texts are to allow different data types. A key-text pair is always
          (key[i], text[i]).
        texts: Numpy array of texts. Same length as keys.
        shard_length: How many pairs are in a shard. ResumeKTI autosaves
          progress after finishing a shard.
        kt_filename: File where to save shard_length, keys, texts. If the file
          exists prints a message, but does not overwrite.
        progress_filename: The file with the number of traces captured so far.
        allow_pickle: Parameter for numpy.save.
    """
    # Check keys and texts lengths.
    assert len(keys) == len(texts)
    assert len(keys) >= 1
    assert len(keys) % shard_length == 0
    try:
        with open(kt_filename, "xb") as kt_file:
            np.save(kt_file, shard_length, allow_pickle=allow_pickle)
            np.save(kt_file, keys, allow_pickle=allow_pickle)
            np.save(kt_file, texts, allow_pickle=allow_pickle)
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
                 kt_filename: str = "key_text_pairs.txt",
                 progress_filename: str = "progress_pairs.txt",
                 allow_pickle: bool = False) -> None:
        """Create a resumeable key-text iterable.

        Args:
            kt_filename: File to load shard_length, keys, texts.
            progress_filename: The file with the number of traces captured so
              far.  Gets updated by iterating shard length times or manually
              calling _save_progress.
            allow_pickle: Parameter for numpy.load.
        """
        # Load shard length, keys, and texts.
        with open(kt_filename, "rb") as kt_pairs:
            self._shard_length = np.load(kt_pairs, allow_pickle=allow_pickle)
            self._keys = np.load(kt_pairs, allow_pickle=allow_pickle)
            self._texts = np.load(kt_pairs, allow_pickle=allow_pickle)
        # Check length of keys and texts.
        assert len(self._keys) == len(self._texts)
        assert len(self._keys) >= 1
        assert len(self._keys) % self._shard_length == 0

        # Load previous progress.
        self._progress_filename = progress_filename
        with open(self._progress_filename, encoding="utf-8") as progress_file:
            # Ensure self._index is a multiple of self._shard_length.
            i = int(progress_file.read())
            self._index = i - (i % self._shard_length)
        # Check validity of the progress.
        assert 0 <= self._index <= len(self._keys)
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
            progress_file.write(str(i))

    def __iter__(self):
        return self

    def __next__(self):
        """Next with auto-save.

        Auto-save when a new shard is starting (after each shard length + 1
        calls without a _save_progress call), thus updates content of
        self._progress_filename file.

        Returns:
            Key-text pair.
        """
        if self._index % self._shard_length == 0:
            self._save_progress()
        if self._index >= len(self._keys):
            raise StopIteration
        self._index += 1
        return self._keys[self._index - 1], self._texts[self._index - 1]

    def __len__(self):
        return len(self._keys) - self._initial_index


def create_resume_kti(keys: np.ndarray,
                      texts: np.ndarray,
                      shard_length: np.uint64,
                      kt_filename: str = "key_text_pairs.txt",
                      progress_filename: str = "progress_pairs.txt",
                      allow_pickle: bool = False) -> ResumeKTI:
    """Saves the key-text pairs and returns a ResumeKTI object for immediate
    use.

    Args:
        keys: Numpy array of keys to capture traces with. Non-zero length,
          length is a multiple of shard_length. Different array for keys and
          texts are to allow different data types. A key-text pair is always
          (key[i], text[i]).
        texts: Numpy array of texts. Same length as keys.
        shard_length: How many pairs are in a shard. ResumeKTI autosaves
          progress after finishing a shard.
        kt_filename: File where to save shard_length, keys, texts. If the file
          exists prints a message, but does not overwrite.
        progress_filename: The file with the number of traces captured so far.
        allow_pickle: Parameter for numpy.save.

    Returns: ResumeKTI object just created.
    """
    _create_resume_kti(keys=keys,
                       texts=texts,
                       shard_length=shard_length,
                       kt_filename=kt_filename,
                       progress_filename=progress_filename,
                       allow_pickle=allow_pickle)
    return ResumeKTI(kt_filename=kt_filename,
                     progress_filename=progress_filename,
                     allow_pickle=allow_pickle)
