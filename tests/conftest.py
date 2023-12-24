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
    from e3nn_jax._src.config import __default_conf

    for key, value in __default_conf.items():
        e3nn.config(key, value)

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
