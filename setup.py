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
import os
from pathlib import Path
from setuptools import find_packages
from setuptools import setup
import subprocess
import sys
from time import time

# Keep compatibility with python3 setup.py develop and install also the packages
# that are being split.
def install_sub_packages():
    # List of sub-packages that are being split.
    sub_packages = ["scaaml_dataset"]
    # Path to the scaaml directory.
    repository_path = Path(os.path.realpath(__file__)).parent
    # Install individual sub-packages.
    for package_name in sub_packages:
        package_path = repository_path / "packages" / package_name
        # This is the recommended way:
        # https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--editable", str(package_path)],
            check=True)

install_sub_packages()


with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()
version = f"2.0.1r{int(time())}"

setup(
    name="scaaml",
    python_requires=">=3.7",
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
