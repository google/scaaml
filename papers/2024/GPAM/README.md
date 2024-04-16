# Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning

[Full version](https://arxiv.org/abs/2306.07249)

To make cryptographic processors more resilient against side-channel attacks,
engineers have developed various countermeasures. However, the effectiveness of
these countermeasures is often uncertain, as it depends on the complex interplay
between software and hardware. Assessing a countermeasure’s effectiveness using
profiling techniques or machine learning so far requires significant expertise
and effort to be adapted to new targets which makes those assessments expensive.
We argue that including cost-effective automated attacks will help chip design
teams to quickly evaluate their countermeasures during the development phase,
paving the way to more secure chips.

In this paper, we lay the foundations toward such automated system by proposing
GPAM, the first deep-learning system for power side-channel analysis that
generalizes across multiple cryptographic algorithms, implementations, and
side-channel counter-measures without the need for manual tuning or trace
preprocessing. We demonstrate GPAM’s capability by successfully attacking four
hardened hardware-accelerated elliptic-curve digital-signature implementations.
We showcase GPAM’s ability to generalize across multiple algorithms by attacking
a protected AES implementation and achieving comparable performance to
state-of-the-art attacks, but without manual trace curation and within a limited
budget. We release our data and models as an open-source contribution to allow
the community to independently replicate our results and build on them.

## List of datasets used in this paper

### Newly captured datasets:

| Name    | Trace length | Examples in train | test  | holdout | Download size (in TB) | Download and usage instructions                                           |
| ------- | ------------ | ----------------: | ----- | ------- | --------------------: | ------------------------------------------------------------------------- |
| ECC CM0 | 1,6M         | 57,344            | 8,192 | 8,192   | 0.2                   | [ECC CM0](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM) |
| ECC CM1 | 5M           | 194,544           | 8,192 | 8,192   | 1.5                   | [ECC CM1](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM) |
| ECC CM2 | 10M          | 122,880           | 8,192 | 8,192   | 2.1                   | [ECC CM2](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM) |
| ECC CM3 | 17,5M        | 122,880           | 8,192 | 8,192   | 3.7                   | [ECC CM3](https://github.com/google/scaaml/tree/main/papers/datasets/ECC/GPAM) |

### Publicly available datasets:

[REASSURE (H2020 731591) ECC Dataset](https://zenodo.org/records/3609789), Łukasz Chmielewski

[ASCADv2](https://www.data.gouv.fr/en/datasets/ascadv2/) Loïc Masure and Rémi Strullu.
Side-channel analysis against ANSSI’s protected AES implementation on ARM: end-to-end attacks with multi-task learning.
Journal of Cryptographic Engineering, 2023. [https://eprint.iacr.org/2021/592.pdf]

[ASCADv1](https://github.com/ANSSI-FR/ASCAD) ATMEGA boolean masked AES variable key, Ryad Benadjila, Prouff Emmanuel, and Junwei Wang.

[CHES 2023 SMAesH challenge](https://smaesh-challenge.simple-crypto.org/) Gaëtan Cassiers, Charles Momin, and François-Xavier Standaert.

## Models

TODO

## List of citations and followup work

This section is best effort. We will be happy if you send us a pull request with
an update, open an issue or just send us an email.

- Remove testing print TODO DO NOT SUBMIT

- I am a loong list TODO DO NOT SUBMIT

## Cite as

```bibtex
@article{bursztein2023generic,
  title={Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning},
  author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Moghimi, Daniel and Picod, Jean-Michel and Zhang, Marina},
  journal={arXiv preprint arXiv:2306.07249},
  year={2023}
}
