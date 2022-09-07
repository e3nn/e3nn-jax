import pytest
import jax
import e3nn_jax as e3nn


class _PRNGKey:
    def __init__(self, key):
        self.key = key

    def __next__(self):
        self.key, key = jax.random.split(self.key)
        return key

    def __getitem__(self, i):
        return jax.random.PRNGKey(i)


@pytest.fixture
def keys():
    return _PRNGKey(jax.random.PRNGKey(24))


@pytest.fixture(autouse=True)
def e3nn_config():
    e3nn.config("irrep_normalization", "component")
    e3nn.config("path_normalization", "element")
    e3nn.config("gradient_normalization", "path")
    e3nn.config("spherical_harmonics_algorithm", "automatic")
    e3nn.config("spherical_harmonics_normalization", "component")
