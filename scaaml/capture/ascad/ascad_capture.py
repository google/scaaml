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
"""Capture script for easier manipulation."""

from scaaml.aes_forward import AESSBOX
from scaaml.capture.aes.aes_capture_contexts import capture_aes_dataset
from scaaml.capture.scope import PicoScope

from typing import Any, Dict, Optional, Type


def capture_ascad_default_parameters(
        firmware_sha256: str,
        crypto_implementation: Type[AESSBOX] = AESSBOX,
        algorithm: str = "simpleserial-aes",
        version: int = 1,
        root_path: str = "/mnt/storage/chipwhisperer",
        url: str = "",
        firmware_url: str = "",
        paper_url: str = "",
        licence: str = "https://creativecommons.org/licenses/by/4.0/",
        examples_per_shard: int = 64,
        train_iterator: Optional[dict[str, Any]] = None,
        test_iterator: Optional[dict[str, Any]] = None,
        holdout_iterator: Optional[dict[str, Any]] = None,
        measurements_info: Optional[Dict[str, Any]] = None) -> None:
    """SCAAML STM32F415 ASCAD specific defaults for capture_ascad_dataset."""
    architecture: str = "CW308_STM32F415"
    implementation: str = "ASCAD"
    shortname: str = "stm32f415_hwaes"
    description: str = "SCAAML HWAES"

    # Parameters for the scope
    samples: int = 160_000
    sample_rate: float = 2.5e9  # Hz
    offset: int = 0
    # 0.5V for a differential probe
    # 0.1V for the BNC connector on the board
    trace_probe_range: float = 0.5  # V
    trigger_range: float = 5.0  # V
    trigger_level: float = 1.9  # V
    capture_info = {
        "samples": samples,
        "trigger_level": trigger_level,
        "trigger_range": trigger_range,
        "sample_rate": sample_rate,
        "offset": offset,
        "trace_probe_range": trace_probe_range,
        "train_chip_id": 620510,  # STM32F415 Training
        "holdout_chip_id": 765432,
    }

    capture_aes_dataset(
        scope_class=PicoScope,
        firmware_sha256=firmware_sha256,
        architecture=architecture,
        implementation=implementation,
        shortname=shortname,
        description=description,
        crypto_implementation=crypto_implementation,
        algorithm=algorithm,
        version=version,
        root_path=root_path,
        url=url,
        firmware_url=firmware_url,
        paper_url=paper_url,
        licence=licence,
        examples_per_shard=examples_per_shard,
        measurements_info=measurements_info,
        capture_info=capture_info,
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        holdout_iterator=holdout_iterator,
    )
