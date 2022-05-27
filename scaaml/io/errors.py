# Copyright 2021 Google LLC
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
"""Implement an error to indicate that a scaaml.io.Dataset already exists.

Creating scaaml.io.Dataset should not overwrite existing files. When it could
the constructor needs to raise an error, which should also contain the dataset
directory.
"""

from pathlib import Path


class DatasetExistsError(FileExistsError):
    """Error for signalling that the dataset already exists."""

    def __init__(self, dataset_path: Path) -> None:
        """Represents that the dataset already exists.

        Args:
          dataset_path: The dataset path.
        """
        super().__init__(
            f'Dataset info file exists and would be overwritten. Use instead:'
            f' Dataset.from_config(dataset_path="{dataset_path}")')
        self.dataset_path = dataset_path
