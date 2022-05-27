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
"""Check common misspellings of scaaml dataset info files. The misspellings
are kept separately in order not to pollute auto-complete.
"""

from typing import Iterable


def find_misspellings(words: Iterable[str]) -> None:
    """Checks for common misspellings of keys in scaaml dataset info file.

    Args:
      words: The words to check.

    Raises:
      ValueError: If spell_check_word raises.

    Example use:
      fixed_dict = {
        'licence': 'https://creativecommons.org/licenses/by/4.0/',
        'compression': 'GZIP',
      }
      find_misspellings(fixed_dict.keys())  # Check for misspellings of keys.
    """
    misspellings = [
        {
            'supported': 'licence',
            'unsupported': 'license',
        },
    ]
    for word in words:
        for misspelling in misspellings:
            spell_check_word(word, **misspelling, case_sensitive=False)


def spell_check_word(word: str,
                     supported: str,
                     unsupported: str,
                     case_sensitive: bool = False) -> None:
    """Checks if the word is spelled out as the unsupported, if so raises
    ValueError.

    Args:
      word: The word to check.
      supported: The supported spelling.
      unsupported: The spelling that is not supported.
      case_sensitive: Compare the words case sensitive.

    Raises:
      ValueError: If the word is equal to the unsupported one.
    """
    # Handle case insensitive comparison.
    word_cmp = word
    unsupported_cmp = unsupported
    if not case_sensitive:
        word_cmp = word.lower()
        unsupported_cmp = unsupported.lower()

    # Compare.
    if word_cmp == unsupported_cmp:
        raise ValueError(f'Unsupported spelling ({unsupported}) found, use '
                         f'{supported} instead.')
