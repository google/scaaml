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

from collections import defaultdict
import pytest

import tensorflow as tf

from scaaml.io.utils import ddict
from scaaml.io.utils import dtype_name_to_dtype, dtype_dtype_to_name


def test_ddict_list_adding():
    original = {
        'a': {
            1: [],
            2: [3, 1, 4]
        },
        'b': {},
    }
    d = ddict(value=original, levels=2, type_var=list)
    assert d == original
    original['b'][1] = []
    assert 1 not in d['b']
    original['b'][1].append('hello')
    assert 1 not in d['b']

    # Leaves are copied, not deep-copied.
    original['a'][1].append(42)
    assert d['a'] == original['a']
    d['a'][1].append(1024)
    assert d['a'] == original['a']


def test_ddict_empty_list():
    d = ddict(value=None, levels=1, type_var=list)
    assert d == {}
    assert isinstance(d, defaultdict)
    assert isinstance(d[0], list)
    e = ddict(value=None, levels=2, type_var=list)
    assert e == {}
    assert isinstance(e, defaultdict)
    assert isinstance(e[0], defaultdict)
    assert isinstance(e[1][0], list)


def test_ddict_changing_original():
    original = {
        'A': {
            'a': 1,
            'b': 2
        },
        'C': None,
    }
    d = ddict(value=original, levels=2, type_var=int)
    original['C'] = [1, 2, 3]
    assert d['C'] == {}
    assert isinstance(d['C'], defaultdict)
    assert isinstance(d['C']['something'], int)

    original['A']['a'] = 7
    original['A']['c'] = 5
    assert 'c' not in d['A']
    assert d['A'] == {'a': 1, 'b': 2}

    d['D']['d'] = 42
    assert 'D' not in original


def test_ddict_2t_none_and_values():
    original = {
        'A': {
            'a': 1,
            'b': 2
        },
        'C': None,
    }
    d = ddict(value=original, levels=2, type_var=int)
    assert d['A'] == original['A']
    assert d['C'] == {}
    assert isinstance(d, defaultdict)
    assert isinstance(d['not here'], defaultdict)
    assert isinstance(d['not here']['promise'], int)
    assert isinstance(d['A'], defaultdict)
    assert isinstance(d['C'], defaultdict)


def test_ddict_2t_none():
    d = ddict(value=None, levels=2, type_var=int)
    assert d == {}
    assert isinstance(d, defaultdict)
    assert isinstance(d['not here'], defaultdict)
    assert isinstance(d['not here']['promise'], int)


def test_ddict_1t_none():
    d = ddict(value=None, levels=1, type_var=int)
    assert d == {}
    assert isinstance(d, defaultdict)
    assert isinstance(d['not here'], int)


def test_ddict_doc():
    d = {'a': 1, 'b': 2}
    e = ddict(value=d, levels=1, type_var=int)
    assert e == d
    assert isinstance(e, defaultdict)
    assert isinstance(e['not here'], int)

    D = {'A': {'a': 1, 'b': 2}, 'C': {}}
    E = ddict(value=D, levels=2, type_var=int)
    assert E == D
    assert isinstance(E, defaultdict)
    assert isinstance(E['not here'], defaultdict)
    assert isinstance(E['not here']['really'], int)


def test_dtype_to_name():
    assert tf.float32 == dtype_name_to_dtype("float32")
    assert tf.float16 == dtype_name_to_dtype("float16")
    with pytest.raises(ValueError) as value_error:
        dtype = dtype_name_to_dtype("int64")
    assert "Either float16 or float32 expected, got" in str(value_error.value)


def test_name_to_dtype():
    assert "float32" == dtype_dtype_to_name(tf.float32)
    assert "float16" == dtype_dtype_to_name(tf.float16)
    with pytest.raises(ValueError) as value_error:
        dtype = dtype_dtype_to_name(tf.int64)
    assert "Either tf.float16 or tf.float32 expected, got" in str(
        value_error.value)
