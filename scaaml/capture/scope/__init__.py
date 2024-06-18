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
"""Oscilloscope manipulation wrappers."""

from scaaml.capture.scope.scope_base import AbstractSScope
from scaaml.capture.scope.cw_scope import CWScope
from scaaml.capture.scope.default_cw_scope import DefaultCWScope
from scaaml.capture.scope.lecroy.lecroy import LeCroy
from scaaml.capture.scope.picoscope import PicoScope

__all__ = [
    "AbstractSScope",
    "CWScope",
    "DefaultCWScope",
    "LeCroy",
    "PicoScope",
]
