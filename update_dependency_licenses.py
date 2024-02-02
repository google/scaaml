# Copyright 2024 Google LLC
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
"""Generates the DEPENDENCY_LICENSES file, by extracting and classifying
dependencies."""
import setuptools
import pkg_resources
import re
import tabulate

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeAlias

METADATA_FILES = ["METADATA", "PKG-INFO"]

LICENSE_OVERRIDES = {
    "capstone": "BSD",
    "vispy": "BSD",
    "chipwhisperer":
        "BSD",  # This license is exclusively for the capturing tooling.
}

LICENSE_CATEGORIES = {
    "Apache 2.0": "notice",
    "Apache 2.0 OR MIT": "notice",
    "HPND": "notice",
    "BSD": "notice",
    "MIT": "notice",
    "PSF": "notice",
    "MPLv2.0, MIT": "reciprocal, notice",
    "Public domain": "unencumbered",
}


def GetDependencies() -> List[str]:
    """Extracts dependencies from setup.py"""
    dependencies: List[str] = []

    # pylint set_up fails, setup is the name in setuptools, pylint has an
    # exempt on setUp.
    def setup(**kwargs: Dict[str, Any]) -> None:  # pylint: disable=C0103
        dependencies.extend(kwargs["install_requires"])

    setuptools.setup = setup
    with open("setup.py", encoding="utf-8") as setup_file:
        exec(setup_file.read())  # pylint: disable=W0122
    return dependencies


def GetPackageLicenses(
        package_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the licensing metadata from a Python package."""
    packages = pkg_resources.require(package_name)
    package = packages[0]
    for metadata_file in METADATA_FILES:
        if not package.has_metadata(metadata_file):
            continue
        metadata = package.get_metadata(metadata_file)
        if package_name in LICENSE_OVERRIDES:
            package_license = LICENSE_OVERRIDES[package_name]
        else:
            match = re.search(r"^License:\s*(.*)$", metadata,
                              re.MULTILINE | re.IGNORECASE)
            assert match
            package_license = re.sub(r"\sLicences|\sLicense,?|Version\s", "",
                                     match.group(1))
            package_license = re.sub("-", " ", package_license)
        match = re.search(r"^Home-page:\s*(.*)$", metadata,
                          re.MULTILINE | re.IGNORECASE)
        assert match
        homepage = match.group(1)
        return package_license, homepage
    return None, None


HeaderType: TypeAlias = Tuple[str, str, str, str]


@dataclass(frozen=True)
class PackageInfo:
    """Represents the package metadata we are extracting."""
    package: str
    licence: str
    category: str
    homepage: str

    def __lt__(self, other: "PackageInfo") -> bool:
        return self.licence < other.licence

    def __iter__(self) -> Iterator[str]:
        return iter((self.package, self.licence, self.category, self.homepage))

    @staticmethod
    def get_headers() -> HeaderType:
        # Order and count must match the one in __iter__
        return ("Package", "License", "Category", "Homepage")


def GenerateDependencyLicensesFile() -> None:
    """Generates the DEPENDENCY_LICENSES file."""
    license_data: List[PackageInfo] = []
    for package_name in GetDependencies():
        package_license, homepage = GetPackageLicenses(package_name)
        assert package_license
        if not homepage:
            homepage = ""
        license_category = LICENSE_CATEGORIES.get(package_license, "")
        license_data.append(
            PackageInfo(package_name, package_license, license_category,
                        homepage))
    table = tabulate.tabulate(
        sorted(license_data),
        headers=PackageInfo.get_headers(),
    )
    print(table)
    with open("DEPENDENCY_LICENSES", "w", encoding="utf-8") as f:
        f.write(table)


def Main() -> None:
    GenerateDependencyLicensesFile()


if __name__ == "__main__":
    Main()
