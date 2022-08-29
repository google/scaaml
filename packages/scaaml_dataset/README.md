# SCAAML: Side Channel Attacks Assisted with Machine Learning
![SCAAML banner](https://storage.googleapis.com/scaaml-public/visuals/scaaml-banner.png)

SCAAML (Side Channel Attacks Assisted with Machine Learning) is a deep learning framework dedicated to side-channel attacks.
It is written in Python and run on top of TensorFlow 2.x.
This is the dataset package, see also the capture and machine learning packages: TODO.

## Available components

- TODO

## Install

### Dependencies

To use SCAAML you need to have a working version of [TensorFlow 2.x](https://www.tensorflow.org/install) and a version of Python >=3.7

### SCAAML framework install

#### Development install

1. Clone the repository: `git clone github.com/google/scaaml/`
2. Install the SCAAML package in development mode: `python3 -m pip install --editable .` (short `pip install -e .` or legacy `python setup.py develop`)

#### Package install

`pip install scaaml-dataset`

### Dataset and models

List of available datasets and models to be downloaded (with download
instructions): TODO

## Publications & Citation

Here is the list of publications and talks related to SCAAML. If you use any of
its codebase, models or datasets please cite:

```bibtex
@online{bursztein2019scaaml,
  title={SCAAML:  Side Channel Attacks Assisted with Machine Learning},
  author={Bursztein, Elie and others},
  year={2019},
  publisher={GitHub},
  url={https://github.com/google/scaaml},
}
```

Additionally please also cite the talks and publications that are the most relevant
to your work, so reader can quickly find the right information. Last but not
least, you are more than welcome to add your publication/talk to the list below by making a pull request 😊.

### SCAAML AES tutorial

TODO provide instructions how to use this package.

DEF CON talk that provides a practical introduction to AES deep-learning based side-channel attacks

```bibtex
@inproceedings{burszteindc27,
title={A Hacker Guide To Deep Learning Based Side Channel Attacks},
author={Elie Bursztein and Jean-Michel Picod},
booktitle ={DEF CON 27},
howpublished = {\url{https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/}}
year={2019},
editor={DEF CON}
}
```

## Disclaimer

This is not an official Google product.
