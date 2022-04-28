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

"""Utils for scaaml.io"""

from collections import defaultdict
import hashlib
from typing import Dict, Optional


def ddict(value: Optional[Dict], levels: int, type_var):
    """Returns nested defaultdict of defaultdicts (nesting level based on
    level), which are updated with the value dictionary.

    Args:
      value: The dictionary that holds values needs to be either None or the
        same level as the result. All occurences of None which should be a Dict
        as a value are replaced by an empty defaultdict. Values are copied,
        but not deepcopied.
      levels: Number of defaultdict iterations. Must be non-negative. Level
        zero returns value.
      type_var: The type of the default value.

    Example use:
      >>> d = {'a': 1, 'b': 2}
      >>> # e is defaultdict(int) updated by d
      >>> e = ddict(value=d, levels=1, type_var=int)
      >>> e
      defaultdict(<class 'int'>, {'a': 1, 'b': 2})
      >>> D = {'A': {'a': 1, 'b': 2}, 'C': {}}
      >>> # E is defaultdict(lambda: defaultdict(int)) updated by D
      >>> E = ddict(value=D, levels=2, type_var=int)
      >>> E
      defaultdict(<function ddict.<locals>.<lambda> at 0x7fed09f4c790>,
              {'A': defaultdict(<class 'int'>,
                  {'a': 1, 'b': 2}),
              'C': defaultdict(<class 'int'>, {})})
      >>> f = ddict(value=None, levels=1, type_var=list)
    """
    def empty_dd(levels: int, type_var):
        """Returns the right level of defaultdicts."""
        if levels == 1:
            return defaultdict(type_var)
        return defaultdict(lambda: empty_dd(levels - 1, type_var))

    assert levels >= 0
    if levels == 0:
        return value
    result = empty_dd(levels, type_var)
    if value is not None:
        for k in value:
            result[k] = ddict(value[k], levels=levels - 1, type_var=type_var)
    return result


def sha256sum(filename):
    "compute the sha256 of a given file"
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()
