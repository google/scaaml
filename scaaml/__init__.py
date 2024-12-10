# Copyright 2019 Google LLC
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
"""Software version of the SCAAML library.

Format: MAJOR.MINOR.PATCH (see https://pypi.org/project/semver/ for more
  possibilities)

Usage:
  Dataset.from_config (and thus Dataset.get_dataset) are using 'scaaml_version'
  and raise a ValueError if the dataset has been captured with a higher
  version of scaaml.

  When a dataset is loaded and Dataset._write_config is called again then the
  scaaml_version is updated to the current value.

  semver.compare is used to compare two versions of the SCAAML library.
"""
__version__ = "3.0.2"
