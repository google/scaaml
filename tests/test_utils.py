"""Test scaaml.utils"""

import pytest

import scaaml
from scaaml.utils import comparable_version


def test_comparable_version_bad():
    """Test not valid inputs."""
    with pytest.raises(ValueError) as verror:
        comparable_version('1..1')  # Integers should be separated by dots
    assert 'invalid literal for int() with base 10' in str(verror.value)
    with pytest.raises(ValueError) as verror:
        comparable_version('.1')  # Integers should be separated by dots
    assert 'invalid literal for int() with base 10' in str(verror.value)
    with pytest.raises(ValueError) as verror:
        comparable_version('0x12.1')  # Hex not allowed
    assert 'invalid literal for int() with base 10' in str(verror.value)
    with pytest.raises(ValueError) as verror:
        comparable_version('1.2-alpha')  # Letters not allowed
    assert 'invalid literal for int() with base 10' in str(verror.value)
    with pytest.raises(ValueError) as verror:
        comparable_version('a.b.c')  # Letters not allowed
    assert 'invalid literal for int() with base 10' in str(verror.value)


def test_comparable_version_cmp():
    """Test comparing for ok cases."""
    # Current version is the default
    assert comparable_version() == comparable_version(scaaml.__version__)
    assert comparable_version('1.0') < comparable_version('2.0')
    assert comparable_version('1.0') == comparable_version('1.0')
    assert comparable_version('1.9') < comparable_version('2.0')
    assert comparable_version('1.0') < comparable_version('1.1')
    assert comparable_version('0.9') < comparable_version('1.0')
    assert comparable_version('1.0') < comparable_version('1.0.0')
    assert comparable_version('1.0') < comparable_version('1.0.1')
    assert comparable_version('1') < comparable_version('1.0')
    assert comparable_version('1') < comparable_version('2')
    assert comparable_version('1.2.3') < comparable_version('1.2.04')
    assert comparable_version('001.02.003') < comparable_version('1.2.04')
    assert comparable_version('001.02.003') == comparable_version('1.2.3')


def test_comparable_version_known():
    """Test against known result."""
    ver = '1.0.2'
    ver_list = [1, 0, 2]
    assert comparable_version(ver) == ver_list
