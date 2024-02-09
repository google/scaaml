# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exceptions for attack point iterators."""


class LengthIsInfiniteException(Exception):
    """This exception is raised when the `__len__` function is
    called on an infinite iterator."""


class ListNotPrescribedLengthException(Exception):
    """This exception is raised when one of the List of values doesn't
    have the same length as it was prescribed to. 
    This is only for constant iterators."""
