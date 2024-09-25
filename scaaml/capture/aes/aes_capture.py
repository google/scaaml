# Copyright 2021-2024 Google LLC
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
from typing import Any, Dict, Optional, Type

from scaaml.aes_forward import AESSBOX
from scaaml.capture.scope import CWScope
from scaaml.capture.aes.aes_capture_contexts import capture_aes_dataset


def capture_aes_scald_stm32f4_mbedtls(
        *,
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
        measurements_info: Optional[Dict[str, Any]] = None) -> None:
    """SCALD STM32F4 MBEDTLS specific defaults for capture_aes_dataset."""
    architecture: str = "CW308_STM32F4"
    implementation: str = "MBEDTLS"
    shortname: str = "aes_scald"
    description: str = "SCALD AES"

    # Parameters for the scope
    gain: int = 45
    samples: int = 7_000
    offset: int = 0
    clock: int = 7_372_800
    sample_rate: str = "clkgen_x4"
    capture_info = {
        "gain": gain,
        "samples": samples,
        "offset": offset,
        "clock": clock,
        "sample_rate": sample_rate,
        "train_chip_id": 164019,
        "holdout_chip_id": 314159,
    }

    capture_aes_dataset(
        scope_class=CWScope,
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
    )
