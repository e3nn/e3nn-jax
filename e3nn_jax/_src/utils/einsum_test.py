import jax
from jax.test_util import check_grads

from e3nn_jax._src.utils.einsum import einsum


def test_check_grads(keys):
    jax.config.update("jax_enable_x64", True)
    x, y, z = (
        jax.random.normal(next(keys), (10, 3)),
        jax.random.normal(next(keys), (3, 4)),
        jax.random.normal(next(keys), (4, 4)),
    )

    check_grads(
        lambda x, y: einsum("ij,jk->ik", x, y),
        (x, y),
        2,
        modes=["fwd", "rev"],
        atol=1e-7,
        rtol=1e-7,
    )
    check_grads(
        lambda z: einsum("ii->", z),
        (z,),
        2,
        modes=["fwd", "rev"],
        atol=1e-7,
        rtol=1e-7,
    )
    check_grads(
        lambda z: einsum("ij,ij->", z, z),
        (z,),
        2,
        modes=["fwd", "rev"],
        atol=1e-7,
        rtol=1e-7,
    )

    jax.config.update("jax_enable_x64", False)
