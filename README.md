# e3nn-jax [![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn-jax/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn-jax?branch=main)
### [Documentation](https://e3nn-jax.readthedocs.io/en/latest) [![Documentation Status](https://readthedocs.org/projects/e3nn-jax/badge/?version=latest)](https://e3nn-jax.readthedocs.io/en/latest/?badge=latest)

```python
import e3nn_jax as e3nn

# Create a random array made of a scalar (0e) and a vector (1o)
array = e3nn.normal("0e + 1o", jax.random.PRNGKey(0))

print(array)  
# 1x0e+1x1o [ 1.8160863  -0.75488514  0.33988908 -0.53483534]

# Compute the norms
norms = e3nn.norm(array)
print(norms)
# 1x0e+1x0e [1.8160863  0.98560894]

# Compute the norm of the full array
total_norm = e3nn.norm(array, per_irrep=False)
print(total_norm)
# 1x0e [2.0662997]

# Compute the tensor product of the array with itself
tp = e3nn.tensor_square(array)
print(tp)
# 2x0e+1x1o+1x2e
# [ 1.9041989   0.25082085 -1.3709364   0.61726785 -0.97130704  0.40373924
#  -0.25657722 -0.18037902 -0.18178469 -0.14190137]
```

### :rocket: 44% faster than pytorch*

*Speed comparison done with a full model (MACE) during training (revMD-17) on a GPU (NVIDIA RTX A5000)

Please always check the [ChangeLog](ChangeLog.md) for breaking changes.

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
