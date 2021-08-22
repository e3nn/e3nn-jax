import pytest
import jax


@pytest.fixture
def key():
    return jax.random.PRNGKey(18)


@pytest.fixture
def key1():
    return jax.random.PRNGKey(21)


@pytest.fixture
def key2():
    return jax.random.PRNGKey(22)


@pytest.fixture
def key3():
    return jax.random.PRNGKey(23)
