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

import os
import numpy as np
import pytest

from scaaml.io.resume_kti import create_resume_kti, ResumeKTI

KT_FILENAME = 'parameters_tuples.txt'
PROGRESS_FILENAME = 'progress_tuples.txt'
KEYS = np.array([3, 1, 4, 1, 5, 9, 0, 254, 255, 0])
TEXTS = np.array([2, 7, 1, 2, 8, 0, 255, 254, 1, 0])
SHARD_LENGTH = 2


def save_and_load(parameters, path):
    # Check that the files do not exist now
    assert not os.path.isfile(path / KT_FILENAME)
    assert not os.path.isfile(path / PROGRESS_FILENAME)

    resume_kti = create_resume_kti(parameters=parameters,
                                   shard_length=SHARD_LENGTH,
                                   kt_filename=path / KT_FILENAME,
                                   progress_filename=path / PROGRESS_FILENAME)
    # Check that the files exist now
    assert os.path.isfile(path / KT_FILENAME)
    assert os.path.isfile(path / PROGRESS_FILENAME)

    # Check that the tuples have been loaded correctly
    assert len(resume_kti) == len(next(iter(parameters.values())))
    i = 0
    for current_params in resume_kti:
        for name, value in current_params._asdict().items():
            assert value == parameters[name][i]
        i += 1

    # Check that the shard_length has been loaded correctly
    assert resume_kti._shard_length == SHARD_LENGTH


def test_save_and_load_k_t(tmp_path):
    parameters = {"keys": KEYS, "texts": TEXTS}
    save_and_load(parameters, tmp_path)


def test_save_and_load_k_t_m(tmp_path):
    parameters = {
        "keys": KEYS,
        "masks": np.random.randint(50, size=KEYS.shape, dtype=np.uint8),
        "texts": TEXTS,
    }
    save_and_load(parameters, tmp_path)


def test_save_and_load_k_t_m_different_len(tmp_path):
    parameters = {
        "keys":
            KEYS,
        "masks":
            np.random.randint(50,
                              size=KEYS.shape[0] + SHARD_LENGTH,
                              dtype=np.uint8),
        "texts":
            TEXTS,
    }
    with pytest.raises(ValueError) as len_error:
        save_and_load(parameters, tmp_path)
    assert "There are different number of parameter values." == str(
        len_error.value)


def test_save_and_load_k(tmp_path):
    parameters = {
        "keys": KEYS,
    }
    save_and_load(parameters, tmp_path)


def iterate_for(tmp_path, n):
    # Iterates for n iterations, then resumes
    parameters = {"keys": KEYS, "texts": TEXTS}
    resume_kti1 = create_resume_kti(parameters=parameters,
                                    shard_length=SHARD_LENGTH,
                                    kt_filename=tmp_path / KT_FILENAME,
                                    progress_filename=tmp_path /
                                    PROGRESS_FILENAME)

    iter1 = iter(resume_kti1)

    i = 0
    while i < n:
        current_params = next(iter1)
        for name, value in current_params._asdict().items():
            assert value == parameters[name][i]
        i += 1

    # Iterate through the rest of shards
    resume_kti2 = ResumeKTI(kt_filename=tmp_path / KT_FILENAME,
                            progress_filename=tmp_path / PROGRESS_FILENAME)
    iter2 = iter(resume_kti2)
    if n % SHARD_LENGTH == 0 and 0 < n < len(KEYS):
        # Calling next marks that the shard is done, calling next only
        # SHARD_LENGTH times does not close the first shard (that happens after
        # the SHARD_LENGTH+1 st call of next)
        j = n - SHARD_LENGTH
    else:
        j = n - (n % SHARD_LENGTH)
    while j < len(KEYS):
        current_params = next(iter2)
        for name, value in current_params._asdict().items():
            assert value == parameters[name][j]
        j += 1


def test_partial_iteration(tmp_path):
    for n in range(len(KEYS) + 1):
        iterate_for(tmp_path, n)
        os.remove(tmp_path / KT_FILENAME)
        os.remove(tmp_path / PROGRESS_FILENAME)


def test_does_not_overwrite(tmp_path):
    # Create real saving points.
    parameters = {"keys": KEYS, "texts": TEXTS}
    create_resume_kti(parameters=parameters,
                      shard_length=SHARD_LENGTH,
                      kt_filename=tmp_path / KT_FILENAME,
                      progress_filename=tmp_path / PROGRESS_FILENAME)
    # Attempt to overwrite
    inc_keys = KEYS + 1
    inc_texts = TEXTS + 1
    resume_kti = create_resume_kti(parameters={
        "keys": inc_keys,
        "texts": inc_texts
    },
                                   shard_length=SHARD_LENGTH,
                                   kt_filename=tmp_path / KT_FILENAME,
                                   progress_filename=tmp_path /
                                   PROGRESS_FILENAME)

    # Check that the (key, text) pairs have been loaded correctly
    assert len(resume_kti) == len(KEYS)
    i = 0
    for current_params in resume_kti:
        for name, value in current_params._asdict().items():
            assert value == parameters[name][i]
        i += 1

    # Check that the shard_length has been loaded correctly
    assert resume_kti._shard_length == SHARD_LENGTH
