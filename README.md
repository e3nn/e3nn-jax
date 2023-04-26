# e3nn-jax [![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn-jax/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn-jax?branch=main)

### :rocket: 44% faster than pytorch*

*Speed comparison done with a full model (MACE) during training (revMD-17) on a GPU (NVIDIA RTX A5000)

### [Documentation](https://e3nn-jax.readthedocs.io/en/latest) [![Documentation Status](https://readthedocs.org/projects/e3nn-jax/badge/?version=latest)](https://e3nn-jax.readthedocs.io/en/latest/?badge=latest)

Please always check the [ChangeLog](Changelog.md) for breaking changes.

## Installation

To install the latest released version:
```bash
pip install --upgrade e3nn-jax
```

To install the latest GitHub version:
```bash
pip install git+https://github.com/e3nn/e3nn-jax.git
```

## Need Help?
Ask a question in the [discussions tab](https://github.com/e3nn/e3nn-jax/discussions).

## What is different from the PyTorch version?

The main difference is the presence of the class [`IrrepsArray`](https://e3nn-jax.readthedocs.io/en/latest/api/irreps_array.html).
`IrrepsArray` contains the irreps (`Irreps`) along with the data array.

## Citing
```
@misc{e3nn_paper,
    doi = {10.48550/ARXIV.2207.09453},
    url = {https://arxiv.org/abs/2207.09453},
    author = {Geiger, Mario and Smidt, Tess},
    keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences}, 
    title = {e3nn: Euclidean Neural Networks},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```
