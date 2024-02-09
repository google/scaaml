# Copyright 2022-2024 Google LLC
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
"""Statistics module contains randomness tests to ensure that capture went
well.

ExampleIterator iterates over examples from a given dataset.

APCounter counts how many times does each value of an attack point occur.
"""
from scaaml.stats.ap_checker import APChecker
from scaaml.stats.ap_counter import APCounter
from scaaml.stats.example_iterator import ExampleIterator
from scaaml.stats.print_stats import PrintStats
from scaaml.stats.trace_stddev_of_stat import STDDEVofAVGofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofMAXofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofMINofTraces
from scaaml.stats.trace_stddev_of_stat import STDDEVofSTATofTraces

__all__ = [
    "APChecker", "APCounter", "ExampleIterator", "PrintStats",
    "STDDEVofAVGofTraces", "STDDEVofMAXofTraces", "STDDEVofMINofTraces",
    "STDDEVofSTATofTraces"
]
