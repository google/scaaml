# Copyright 2025 Google LLC
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
"""Configure pytest. Mainly skipping slow tests which are not expected to
change much.
https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow to run",
    )


def pytest_collection_modifyitems(config, items):
    # Reason why we skipped the slow tests.
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    # --runslow given in cli: do not skip slow tests
    should_skip_slow: bool = not bool(config.getoption("--runslow"))

    # Still loop to allow adding more markers.
    for item in items:
        # Skip slow tests.
        if "slow" in item.keywords and should_skip_slow:
            item.add_marker(skip_slow)
