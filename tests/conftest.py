import pytest
import jax


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
