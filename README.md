# SCAAML: Side Channel Attacks Assisted with Machine Learning

![SCAAML banner](https://storage.googleapis.com/scaaml-public/visuals/scaaml-banner.png)

SCAAML (Side Channel Attacks Assisted with Machine Learning) is a deep learning
framework dedicated to side-channel attacks. It is written in python and run on
top of TensorFlow 2.x.

[![Coverage Status](https://coveralls.io/repos/github/google/scaaml/badge.svg?branch=main)](https://coveralls.io/github/google/scaaml?branch=main)

## Latest Updates

-   Sep 2024: [GPAM](https://github.com/google/scaaml/tree/main/papers/2024/GPAM)
    the first power side-channel general model capable of attacking multiple
    algorithms using full traces, were presented at CHES and are now available for
    download.

-   Sep 2024: [ECC datasets](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM)
    our large-scale ECC datasets are available for download.

## Available components

-   [`scaaml/`](https://github.com/google/scaaml/tree/master/scaaml/): The
    SCAAML framework code. Its used by the various tools.

-   [`scaaml_intro/`](https://github.com/google/scaaml/tree/master/scaaml_intro):
    *A Hacker Guide To Deep Learning Based Side Channel Attacks*.  Code, dataset
    and models used in our step by step tutorial on how to use deep-learning to
    perform AES side-channel attacks in practice.

-   [`GPAM`](https://github.com/google/scaaml/tree/main/papers/2024/GPAM)
    *Generalized Power Attacks against Crypto Hardware using Long-Range Deep
    Learning* model and datasets needed to reproduce our results are available
    for download.

-   [`ECC datasets`](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM)
    A collection of large-scale hardware protected ECC datasets.

## Install

### Dependencies

To use SCAAML you need to have a working version of [TensorFlow
2.x](https://www.tensorflow.org/install) and a version of Python >=3.9

### SCAAML framework install

1.  Clone the repository: `git clone github.com/google/scaaml/`
2.  Create and activate Python virtual environment:
       `python3 -m venv my_env`
       `source my_env/bin/activate`
3.  Install dependencies: `python3 -m pip install --require-hashes -r
    requirements.txt`
4.  Install the SCAAML package: `python setup.py develop`

## Publications & Citation

Here is the list of publications and talks related to SCAAML. If you use any of
its codebase, models or datasets please cite the repo and the relevant papers:

```bibtex
@software{scaaml_2019,
    title = {{SCAAML: Side Channel Attacks Assisted with Machine Learning}},
    author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Picod, Jean-Michel},
    url = {https://github.com/google/scaaml},
    version = {1.0.0},
    year = {2019}
}
```

## Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning

```bibtex
@article{bursztein2023generic,
  title={Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning},
  author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Moghimi, Daniel and Picod, Jean-Michel and Zhang, Marina},
  journal={CHES},
  year={2024}
}
```

## SCAAML AES tutorial

DEF CON talk that provides a practical introduction to AES deep-learning based
side-channel attacks

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
