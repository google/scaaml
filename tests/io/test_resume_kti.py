import os
import numpy as np

from scaaml.io.resume_kti import create_resume_kti, ResumeKTI

KT_FILENAME = 'key_text_pairs.txt'
PROGRESS_FILENAME = 'progress_pairs.txt'
KEYS = np.array([3, 1, 4, 1, 5, 9, 0, 254, 255, 0])
TEXTS = np.array([2, 7, 1, 2, 8, 0, 255, 254, 1, 0])
SHARD_LENGTH = 2


def test_save_and_load(tmp_path):
    # Check that the files do not exist now
    assert not os.path.isfile(tmp_path / KT_FILENAME)
    assert not os.path.isfile(tmp_path / PROGRESS_FILENAME)
    resume_kti = create_resume_kti(keys=KEYS,
                                   texts=TEXTS,
                                   shard_length=SHARD_LENGTH,
                                   kt_filename=tmp_path / KT_FILENAME,
                                   progress_filename=tmp_path /
                                   PROGRESS_FILENAME)
    # Check that the files exist now
    assert os.path.isfile(tmp_path / KT_FILENAME)
    assert os.path.isfile(tmp_path / PROGRESS_FILENAME)

    # Check that the (key, text) pairs have been loaded correctly
    assert len(resume_kti) == len(KEYS)
    i = 0
    for k, t in resume_kti:
        assert k == KEYS[i]
        assert t == TEXTS[i]
        i += 1

    # Check that the shard_length has been loaded correctly
    assert resume_kti._shard_length == SHARD_LENGTH


def iterate_for(tmp_path, n):
    # Iterates for n iterations, then resumes
    resume_kti1 = create_resume_kti(keys=KEYS,
                                    texts=TEXTS,
                                    shard_length=SHARD_LENGTH,
                                    kt_filename=tmp_path / KT_FILENAME,
                                    progress_filename=tmp_path /
                                    PROGRESS_FILENAME)

    iter1 = iter(resume_kti1)

    i = 0
    while i < n:
        k, t = next(iter1)
        assert k == KEYS[i]
        assert t == TEXTS[i]
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
        k, t = next(iter2)
        assert k == KEYS[j]
        assert t == TEXTS[j]
        j += 1


def test_partial_iteration(tmp_path):
    for n in range(len(KEYS) + 1):
        iterate_for(tmp_path, n)
        os.remove(tmp_path / KT_FILENAME)
        os.remove(tmp_path / PROGRESS_FILENAME)


def test_does_not_overwrite(tmp_path):
    # Create real saving points.
    create_resume_kti(keys=KEYS,
                      texts=TEXTS,
                      shard_length=SHARD_LENGTH,
                      kt_filename=tmp_path / KT_FILENAME,
                      progress_filename=tmp_path / PROGRESS_FILENAME)
    # Attempt to overwrite
    inc_keys = KEYS + 1
    inc_texts = TEXTS + 1
    resume_kti = create_resume_kti(keys=inc_keys,
                                   texts=inc_texts,
                                   shard_length=SHARD_LENGTH,
                                   kt_filename=tmp_path / KT_FILENAME,
                                   progress_filename=tmp_path /
                                   PROGRESS_FILENAME)

    # Check that the (key, text) pairs have been loaded correctly
    assert len(resume_kti) == len(KEYS)
    i = 0
    for k, t in resume_kti:
        assert k == KEYS[i]
        assert t == TEXTS[i]
        i += 1

    # Check that the shard_length has been loaded correctly
    assert resume_kti._shard_length == SHARD_LENGTH
