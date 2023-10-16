# SCAAML: Side Channel Attacks Assisted with Machine Learning
![SCAAML banner](https://storage.googleapis.com/scaaml-public/visuals/scaaml-banner.png)

SCAAML (Side Channel Attacks Assisted with Machine Learning) is a deep learning framework dedicated to side-channel attacks.
It is written in python and run on top of TensorFlow 2.x.

## Available components

- [`scaaml/`](https://github.com/google/scaaml/tree/master/scaaml/): The SCAAML framework code. Its used by the various tools.
- [`scaaml_intro/`](https://github.com/google/scaaml/tree/master/scaaml_intro): *A Hacker Guide To Deep Learning Based Side Channel Attacks*.
  Code, dataset and models used in our step by step tutorial on how to use deep-learning to perform AES side-channel attacks in practice.

## Install

### Dependencies

To use SCAAML you need to have a working version of [TensorFlow 2.x](https://www.tensorflow.org/install) and a version of Python >=3.6


### SCAAML framework install

1. Clone the repository: `git clone github.com/google/scaaml/`
2. Create and activate Python virtual environment:
      `python3 -m venv my_env`
      `source my_env/bin/activate`
3. Install dependencies: `python3 -m pip install --require-hashes -r requirements.txt`
4. Install the SCAAML package: `python setup.py develop`

### Update dependencies

Make sure to have: `sudo apt install python3 python3-pip python3-venv` and
activated the virtual environment.

Install requirements: `pip install --require-hashes -r base-tooling-requirements.txt`

Update: `pip-compile --allow-unsafe requirements.in --generate-hashes --upgrade` and commit requirements.txt.

### Dataset and models

Every SCAAML component rely on a datasets and optional models that you will need to download in the component directory. The link to download those are available in the components specific README.md. Simply click on the directory representing the component of your choice, or the link to the component in the list above.

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

To cite the [paper](https://arxiv.org/abs/2306.07249) describing the approach, please cite:
```bibtex
@misc{bursztein2023generic,
      title={Generic Attacks against Cryptographic Hardware through Long-Range Deep Learning}, 
      author={Elie Bursztein and Luca Invernizzi and Karel Král and Daniel Moghimi and Jean-Michel Picod and Marina Zhang},
      year={2023},
      eprint={2306.07249},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

### SCAAML AES tutorial

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
