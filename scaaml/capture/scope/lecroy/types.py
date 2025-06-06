# Copyright 2023 Google LLC
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
"""Type definitions for LeCroy probes.
"""

from typing import Literal
from typing_extensions import TypeAlias

LeCroyCommunicationClassName: TypeAlias = Literal[
    "LeCroyCommunicationVisa",
    "LeCroyCommunicationSocket",
]

LeCroyCaptureArea: TypeAlias = Literal[
    "FULLSCREEN",
    "GRIDAREAONLY",
    "DSOWINDOW",
]

LeCroyDigChannelName: TypeAlias = Literal[
    "DIGITAL1",
    "DIGITAL2",
    "DIGITAL3",
    "DIGITAL4",
]

LeCroyChannelName: TypeAlias = Literal[
    "C1",
    "C2",
    "C3",
    "C4",
] | LeCroyDigChannelName

LeCroyDigLine: TypeAlias = Literal[
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
]
