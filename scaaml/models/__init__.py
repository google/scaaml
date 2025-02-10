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
"""Published models.

This is a model which can be imported from the SCAAML pip package and is
reasonably up to date. For the original model (archived) see papers/2024/GPAM/

For the [DEF CON
27](https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/)
presentation see scaaml_intro/ for historical code.
"""

from scaaml.models.gpam import get_gpam_model

__all__ = [
    "get_gpam_model",
]
