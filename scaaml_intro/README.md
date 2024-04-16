# SCAAML AES side-channel attacks tutorial

This directory provides the code, models and the dataset needed to reproduce the
AES deep-learning side channel attack demonstrated at
[DEF CON 27](https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/)
and in our online tutorial.

## Important notes

This dataset and code is **for educational and demo purpose only**.  It is not
suitable for research as TinyAES is an easy target and we don't provide the
holdout dataset to do proper evaluations.

If you use this code, models or dataset in any shape or form please provide
attribution by citing our
[DEF CON presentation](https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/)
or having a link to this [repository](https://github.com/google/scaaml).

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

For research purpose you should instead use, when they will be available, our
large scale benchmark datasets which have a wide variety of targets with
different levels of difficulty and holdout datasets made on different hardware
board to test model generalization.  We will announce it on Twitter, blog and
other channels when available.

Finally the models are significantly smaller and yield better results than the
original presentation as they were updated to use one of our latest model
architecture which is significantly more efficient.

## Setup

### Framework install

In order to run the notebooks/train models you need to install the SCAAML
framework as described in the main [README](https://github.com/google/scaaml/)

### Dataset & models

In order to run the notebooks/train models you need to download the following
dataset and models:

| Filename                                                                               | What it is                                                | Download size | Expected Location | SHAS256                                                          |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------- | :-----------: | ----------------- | ---------------------------------------------------------------- |
| [datasets.zip](https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip) | TinyAES train & test datasets                             |     8.2GB     | `datasets/`       | 4bf2c6defb79b40b30f01f488e83762396b56daad14a694f64916be2b665b2f8 |
| [models.zip](https://storage.googleapis.com/scaaml-public/scaaml_intro/models.zip)     | TinyAES 48 pretrained models - 3 attack points * 16 bytes |     312MB     | `models/`         | 17d7d32cca0ac0db157ae1f5696f6c64bba6d753a8f33802d0d9614bb07d3d9b |
| [logs.zip](https://storage.googleapis.com/scaaml-public/scaaml_intro/logs.zip)         | Tensorboard training logs (optional)                      |     616MB     | `logs`            | 5b2f43f89990653d64820cca61f15fc6818ee674ae4cc2b4f235cfd9a48f3b28 |

Make sure to unzip them in this directory (`scaaml_demo`) otherwise the code
won't find them.

Note: the Tensorboard logs are optional, they are mostly provided for people
interested in looking at how fast the models converged.

## Usage

The code is split into two parts:

-   `train.py` is used to train the attack models. It takes as argument a config
    that defines what to train. The configuration used in the tutorial is
    located here `config/stm32f415_tinyaes.json` and you can use it to train
    your own models by running
    `python train.py -c config/stm32f415_tinyaes.json`

-   `key_recovery_demo.ipynb` is the notebook that showcases how to use the
    trained model and the `scaaml` library to recover TinyAES keys with just 4
    traces. You can either use the provided models, or train your own.
