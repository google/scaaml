# Copyright 2026 Google LLC
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
"""Correlation power analysis (CPA) module. This module is structured to
support multiple backends (e.g., a GPU accelerated JAX implementation and a
NumPy implementation). If JAX is installed defaults to JAX implementation
otherwise falls back to the NumPy one.

If a concrete version is needed use:
```python
from scaaml.stats.cpa.cpa import CPA
# from scaaml.stats.cpa.cpa_jax import CPA
```
"""
from importlib.metadata import PackageNotFoundError

try:
    # JAX based if JAX is installed
    from scaaml.stats.cpa.cpa_jax import CPA
except PackageNotFoundError:
    # NumPy based default
    from scaaml.stats.cpa.cpa import CPA  # type: ignore[assignment]

__all__ = [
    "CPA",
]
