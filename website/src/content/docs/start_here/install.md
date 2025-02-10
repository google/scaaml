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
downside of no having these libraries is some failing unit-tests.  Follow the
official guide
[https://www.picotech.com/downloads/linux](https://www.picotech.com/downloads/linux)
or on a Debian based system one can just:

```bash
sh -c 'wget -qO - https://labs.picotech.com/Release.gpg.key | sudo apt-key add -'
sudo sh -c 'echo "deb https://labs.picotech.com/picoscope7/debian/ picoscope main" >/etc/apt/sources.list.d/picoscope7.list'
apt-get update && apt-get install -y udev usbutils
mv /sbin/udevadm /sbin/udevadm.bin ; echo '#!/bin/bash' > /sbin/udevadm && chmod a+x /sbin/udevadm
apt-get update && apt-get install -y libps6000a
# apt-get update && apt-get install -y picoscope  # Full install with GUI and mono
```

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
