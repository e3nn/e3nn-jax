from e3nn_jax._src.util.jit import jit_code


def test_jit_code():
    assert "add" in jit_code(lambda x: x + 1, 1.2)
