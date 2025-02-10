---
title: Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning
description: Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning
---

At CHES 2024 we presented our generic model aimed to be used for side-channel analysis across algorithms without preprocessing.

# Abstract

To make cryptographic processors more resilient against side-channel attacks, engineers have developed various countermeasures.
However, the effectiveness of these countermeasures is often uncertain, as it depends on the complex interplay between software and hardware.
Assessing a countermeasure's effectiveness using profiling techniques or machine learning so far requires significant expertise and effort to be adapted to new targets which makes those assessments expensive.
We argue that including cost-effective automated attacks will help chip design teams to quickly evaluate their countermeasures during the development phase, paving the way to more secure chips.

In this paper, we lay the foundations toward such automated system by proposing GPAM, the first deep-learning system for power side-channel analysis that generalizes across multiple cryptographic algorithms, implementations, and side-channel countermeasures without the need for manual tuning or trace preprocessing.
We demonstrate GPAM's capability by successfully attacking four hardened hardware-accelerated elliptic-curve digital-signature implementations.
We showcase GPAM's ability to generalize across multiple algorithms by attacking a protected AES implementation and achieving comparable performance to state-of-the-art attacks, but without manual trace curation and within a limited budget.
We release our data and models as an open-source contribution to allow the community to independently replicate our results and build on them.

# References

## Cite as

```bibtex
@article{bursztein2023generic,
  title={Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning},
  author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Moghimi, Daniel and Picod, Jean-Michel and Zhang, Marina},
  journal={IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume={2024},
  number={3},
  pages={472--499},
  year={2024}
}
```

## Media

- `from scaaml.models import get_gpam_model` (with `scaaml>=3.0.3`).
- [Archived version of code, datasets, and pretrained models](https://github.com/google/scaaml/tree/main/papers/2024/GPAM).
  Beware that the saved models were saved using Keras2 while the current version is Keras3.
  A safe way to be compatible is to `python -m pip install "tensorflow==2.12"`.
- [Recording of all talks from the block](https://youtu.be/qDuamuHPwlk)
- [Slides (online)](https://docs.google.com/presentation/d/1Wi8eTE-d1CALF9R-EqaatVVrLD9lWFLFXwXe31Rzlkk/embed?start=false&loop=false&delayms=3000)
- [Slides (pdf)](https://iacr.org/submit/files/slides/2024/tches/tches2024/3_71/3_71_slides.pdf)
- [arXiv](https://arxiv.org/abs/2306.07249)
- [TCHES version of the paper](https://tches.iacr.org/index.php/TCHES/article/view/11685/11205)
