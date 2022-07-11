# Copyright 2020 Google LLC
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
"""Setup script."""
from setuptools import find_packages
from setuptools import setup
from time import time

with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()
version = f"2.0.1r{int(time())}"

setup(
    name="scaaml",
    python_requires='>=3.6',
    version=version,
    description="Side Channel Attack Assisted with Machine Learning",
    long_description=long_description,
    author="Elie Bursztein",
    author_email="scaaml@google.com",
    url="https://github.com/google/scaaml",
    license="Apache License 2.0",
    install_requires=[
        "colorama",
        "termcolor",
        "tqdm",
        "pandas",
        "pytest",
        "numpy",
        "tabulate",
        "matplotlib",
        "Pillow",
        "tensorflow>=2.2.0",
        "pygments",
        "chipwhisperer",
        "scipy",
        "semver",
    ],
    package_data={"": ["*.pickle"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Framework :: Jupyter",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
)
