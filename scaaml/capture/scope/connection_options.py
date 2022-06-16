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
"""Connection options for `scope.con` method."""

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class ConnectionOptions:
    sn: Optional[str] = None
    idProduct: Optional[int] = None  # pylint: disable=C0103
    bitstream: Optional[str] = None
    force: bool = False
    prog_speed: float = 10E6
    kwargs: Optional[Any] = None
