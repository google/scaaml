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
"""Type that is both OpenADC and a ScopeTemplate."""

from chipwhisperer.capture.scopes import OpenADC

from scaaml.capture.scope.scope_template import ScopeTemplate


class ChipwhispererScope(ScopeTemplate, OpenADC):
    """A base class for scope objects that can be passed as a scope to
    chipwhisperer API."""
