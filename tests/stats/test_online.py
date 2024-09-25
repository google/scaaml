# Copyright 2024 Google LLC
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

import numpy as np

from scaaml.stats.online import *


def test_sum_easy():
    s = Sum()

    s.update(1)
    s.update(2)
    s.update(3)

    assert s.result == 6


def test_sum_large_and_small():
    s = Sum()

    s.update(1.0)
    s.update(10**100)
    s.update(1.0)
    s.update(-10**100)

    assert s.result == 2


def test_sum_docstring_example():
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = Sum()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.sum(axis=0))


def test_mean():
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = Mean()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.mean(axis=0))


def test_mean():
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = Mean()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.mean(axis=0))


def test_variance_single_pass_ddof0():
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = VarianceSinglePass()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.var(axis=0))


def test_variance_single_pass_ddof1():
    # For ddof test it is good to have small first dimension.
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = VarianceSinglePass(ddof=1)

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.var(axis=0, ddof=1))


def test_variance_two_pass_ddof0():
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = VarianceTwoPass()

    for e in a:
        s.update(e)

    s.set_second_pass()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.var(axis=0))


def test_variance_two_pass_ddof1():
    # For ddof test it is good to have small first dimension.
    a = np.random.uniform(0.0, 1.0, (20, 100))
    s = VarianceTwoPass(ddof=1)

    for e in a:
        s.update(e)

    s.set_second_pass()

    for e in a:
        s.update(e)

    np.testing.assert_allclose(s.result, a.var(axis=0, ddof=1))
