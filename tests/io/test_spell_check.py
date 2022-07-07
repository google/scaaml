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
"""Unit tests of scaaml/io/spell_check.py"""

import pytest

from scaaml.io.spell_check import find_misspellings, spell_check_word


def test_find_misspellings_bad():
    """Misspelling to be found."""
    bad_words = ['lorem', 'licence', 'dolor', 'sit', '', 'license']
    with pytest.raises(ValueError) as value_error:
        find_misspellings(words=bad_words)
    assert 'Unsupported spelling' in str(value_error.value)


def test_find_misspellings_with_dictionary():
    dictionary = {
        'licence': 'https://creativecommons.org/licenses/by/4.0/',
        'compression': 'GZIP',
    }
    find_misspellings(words=dictionary.keys())

    dictionary['license'] = 'Wrong spelling'
    with pytest.raises(ValueError) as value_error:
        find_misspellings(words=dictionary.keys())
    assert 'Unsupported spelling' in str(value_error.value)


def test_find_misspellings_ok():
    """No misspelling."""
    ok_words = ['lorem', 'ipsum', 'dolor', 'sit', '', 'licence']
    find_misspellings(words=ok_words)


def test_spell_check_word():
    # ok
    spell_check_word(word='licence',
                     supported='licence',
                     unsupported='license',
                     case_sensitive=False)
    # ok
    spell_check_word(word='licence',
                     supported='licence',
                     unsupported='license',
                     case_sensitive=True)
    # ok (case sensitive test)
    spell_check_word(
        word='license',
        supported='licence',
        unsupported='LICENSE',  # Not equal
        case_sensitive=True)
    with pytest.raises(ValueError) as value_error:
        # raise
        spell_check_word(
            word='license',
            supported='licence',
            unsupported='LICENSE',  # Not equal
            case_sensitive=False)
    assert 'Unsupported spelling' in str(value_error.value)
