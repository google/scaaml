---
title: SCAAML Installation
description: SCAAML software installation
---

Installation of the SCAAML Python package.

## Dependencies

### [Optional] PicoScope &reg; 6424E

To use the Python oscilloscope control of a [PicoScope
&reg;](https://www.picotech.com/products/oscilloscope) 6424E one needs to
install the corresponding libraries.  If you do not plan to use this the only
downside of not having these libraries is that some unit-tests will fail.
Follow the official guide
[https://www.picotech.com/downloads/linux](https://www.picotech.com/downloads/linux).

## Python Package Index

All one should need to install the package from
[PyPI](https://pypi.org/project/scaaml/) is:

```bash
pip install scaaml
```

Note that this is the latest stable version of the package.
If you want the bleeding edge features install from source instead.

## Installing from Source

One can always opt for installation from the source.

```bash
git clone github.com/google/scaaml/  # Clone the repository
python3 -m venv my_env  # Create Python virtual environment
source my_env/bin/activate  # Activate your virtual environment
cd scaaml/  # Change directory to the cloned git repository
python3 -m pip install --require-hashes -r requirements.txt  # Install dependencies
python3 -m pip install --editable .  # Install SCAAML
```
