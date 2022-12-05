# Copyright 2021 Google LLC
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
"""Generating uniformly distributed keys."""

import random
from typing import Iterable, Optional

import numpy as np
from chipwhisperer.capture.acq_patterns._base import AcqKeyTextPattern_Base


class AcqKeyTextPatternScaaml(AcqKeyTextPattern_Base):
    """Class for getting uniformly distributed keys and plaintexts for SCAAML.

       Typical usage example:
         ktp = AcqKeyTextPatternScaaml()
         ktp.dataset = "Training"
         ktp.nb_keys = 3072
         ktp.plaintext_per_key = 256
         kpt.repetitions = 1
         ktp.init(0)
         key, text = ktp.new_pair()
    """
    _name = "SCAAML"
    DATASET_TRAINING = "Training"
    DATASET_VALIDATION = "Validation"
    DATASET_TYPES = (DATASET_TRAINING, DATASET_VALIDATION)

    def __init__(self):
        AcqKeyTextPattern_Base.__init__(self)
        self._pt_per_key = 256
        self._repeat = 4
        self._nbkeys = 3072

        self._dataset = self.DATASET_TRAINING
        self._input_generator: Optional[Iterable] = None
        self._input_shape = (self._key_len, self._text_len)

        self._key = bytearray(
            [random.randint(0, 255) for _ in range(self._key_len)])
        self._textin = bytearray(
            [random.randint(0, 255) for _ in range(self._text_len)])

    @property
    def key_len(self):
        return self._key_len

    @key_len.setter
    def key_len(self, n):
        self._key_len = n
        self._input_generator = None

    @property
    def text_len(self):
        return self._text_len

    @text_len.setter
    def text_len(self, n):
        self._text_len = n
        self._input_generator = None

    @property
    def plaintext_per_key(self):
        """How many different plaintexts are used with a single key."""
        return self._pt_per_key

    @plaintext_per_key.setter
    def plaintext_per_key(self, val):
        if self._dataset == self.DATASET_TRAINING:
            if val % 256:
                raise ValueError("plaintext_per_key must be a multiple of 256")
            self._pt_per_key = val
        elif self._dataset == self.DATASET_VALIDATION:
            self._pt_per_key = val
        else:
            raise ValueError("Unsupported dataset type")

    @property
    def repetitions(self):
        """How many times is a concrete (key, plaintext) pair repeated."""
        return self._repeat

    @repetitions.setter
    def repetitions(self, val):
        """How many times is a concrete (key, plaintext) pair repeated."""
        self._repeat = val

    @property
    def dataset(self):
        """dataset type"""
        return self._dataset

    @dataset.setter
    def dataset(self, val):
        """set dataset type"""
        if val not in self.DATASET_TYPES:
            raise ValueError(f"dataset must be {self.DATASET_TYPES}")
        self._dataset = val

    @property
    def nb_keys(self):
        """nb_keys"""
        return self._nbkeys

    @nb_keys.setter
    def nb_keys(self, val):
        if self._dataset == self.DATASET_TRAINING:
            if val % 256:
                raise ValueError("nb_keys must be a multiple of 256")
            self._nbkeys = val
        elif self._dataset == self.DATASET_VALIDATION:
            self._nbkeys = val
        else:
            raise ValueError("Unsupported dataset type")

    def init(self, maxtraces: int):
        del maxtraces  # unused
        if self._input_shape != (self.key_len, self.text_len):
            # Force recreating the generator if shape has changed. This
            # shouldn't happen while generating a given dataset but may happen
            # when generating different datasets for different algorithms using
            # the GUI.
            self._input_generator = None
        if not self._input_generator:
            if self._dataset == self.DATASET_TRAINING:
                self._input_generator = self._get_dataset_generator()
            elif self._dataset == self.DATASET_VALIDATION:
                self._input_generator = self._get_random_generator()
            else:
                raise ValueError("Unsupported dataset")

    init_pair = init

    def _generate_inputs(self):

        def test_matrix(matrix, length, num_batch=1):
            values, counts = np.unique(matrix, return_counts=True)
            if len(values) != 256:
                return False
            return (counts == (length * num_batch)).all()

        def generate_matrix(length):
            """Generate a length x 256 matrix that spread values with no
            repetition."""
            matrix = np.zeros((length, 256),
                              dtype=np.uint8)  # make the first matrix fail test
            bytes_ = np.arange(256, dtype=np.uint8)

            while not test_matrix(matrix, length):
                for i in range(length):
                    np.random.shuffle(bytes_)
                    matrix[i] = bytes_[:]
            return np.transpose(matrix)

        keys = generate_matrix(self.key_len)
        num_batch = self._pt_per_key // 256
        pts = []
        for _ in range(256):
            plain_text = generate_matrix(self.text_len)
            for _ in range(num_batch - 1):
                plain_text = np.concatenate(
                    (plain_text, generate_matrix(self.text_len)))
            pts.append(np.transpose(plain_text))
        return keys, np.transpose(pts, (0, 2, 1))

    def _get_dataset_generator(self):
        self._input_shape = (self.key_len, self.text_len)
        for _ in range(0, self._nbkeys, 256):
            keys, pts = self._generate_inputs()
            for key_idx, key_value in enumerate(keys):
                for plain_text in pts[key_idx]:
                    for _ in range(self._repeat):
                        yield bytearray(key_value), bytearray(plain_text)

    def _get_random_generator(self):
        for _ in range(self._nbkeys):
            key = bytearray(
                [random.randint(0, 255) for _ in range(self.key_len)])
            for _ in range(self._pt_per_key):
                plain_text = bytearray(
                    [random.randint(0, 255) for _ in range(self.text_len)])
                for _ in range(self._repeat):
                    yield key, plain_text

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self):
        """Convenience for calling new_pair (allow to use self in a for loop).
        """
        return self.new_pair()

    def new_pair(self):
        assert self._input_generator is not None
        self._key, self._textin = next(self._input_generator)
        if self._dataset == self.DATASET_TRAINING:
            if self._pt_per_key % 256:
                raise ValueError("plaintext_per_key must be a multiple of 256")
            if self._nbkeys % 256:
                raise ValueError("nb_keys must be a multiple of 256")
        elif self._dataset == self.DATASET_VALIDATION:
            # Nothing to check
            pass
        else:
            raise ValueError(f"Incorrect dataset type. "
                             f"Allowed values are {self.DATASET_TYPES}. "
                             f"Currently set to {self._dataset}")

        # Check pair works with target
        self.validateKey()
        self.validateText()

        return self._key, self._textin

    def __str__(self):
        return (f"{self._name} ({self._dataset}, {self._pt_per_key}, "
                f"{self._repeat}, {self._nbkeys})")
