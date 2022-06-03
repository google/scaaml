"""Generates the DEPENDENCY_LICENSES file, by extracting and classifying
dependencies."""
import setuptools
import pkg_resources
import re
import tabulate

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


def GetDependencies():
    """Extracts dependencies from setup.py"""
    dependencies = []

    # pylint set_up fails, setup is the name in setuptools, pylint has an
    # extempt on setUp.
    def setup(**kwargs):  # pylint: disable=C0103
        dependencies.extend(kwargs["install_requires"])

    setuptools.setup = setup
    with open("setup.py", encoding="utf-8") as setup_file:
        exec(setup_file.read())  # pylint: disable=W0122
    return dependencies


def GetPackageLicenses(package_name):
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
            package_license = re.sub(r"\sLicences|\sLicense,?|Version\s", "",
                                     match.group(1))
            package_license = re.sub("-", " ", package_license)
        match = re.search(r"^Home-page:\s*(.*)$", metadata,
                          re.MULTILINE | re.IGNORECASE)
        homepage = match.group(1)
        return package_license, homepage
    return None


def GenerateDependencyLicensesFile():
    """Generates the DEPENDENCY_LICENSES file."""
    license_data = []
    for package_name in GetDependencies():
        package_license, homepage = GetPackageLicenses(package_name)
        license_category = LICENSE_CATEGORIES.get(package_license, "")
        license_data.append(
            (package_name, package_license, license_category, homepage))
    table = tabulate.tabulate(
        sorted(license_data, key=lambda x: x[1]),
        headers=["Package", "License", "Category", "Homepage"],
    )
    print(table)
    with open("DEPENDENCY_LICENSES", "w", encoding="utf-8") as f:
        f.write(table)


def Main():
    GenerateDependencyLicensesFile()


if __name__ == "__main__":
    Main()
