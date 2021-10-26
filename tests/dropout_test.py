import haiku as hk
import jax
import random
from sys import maxsize

from e3nn_jax import Dropout, Irreps
from e3nn_jax.util.test import assert_equivariant


def test_dropout():
    irreps = Irreps("10x1e + 10x0e")
    rng_key = jax.random.PRNGKey(random.randint(0, maxsize))
    rng_key, *sub_keys = jax.random.split(rng_key, num=7)

    @hk.without_apply_rng
    @hk.transform
    def b(rng, x, is_training=True):
        m = Dropout(irreps=irreps, p=0.75)
        return m(rng, x, is_training)

    params = b.init(rng_key, rng_key, irreps.randn(sub_keys[0], (5, 2, -1)))
    x = irreps.randn(sub_keys[1], (5, 2, -1))

    y = b.apply(params, sub_keys[2], x, is_training=False)
    assert (y == x).all()

    y = b.apply(params, sub_keys[3], x)
    assert ((y == (x / 0.25)) | (y == 0)).all()

    def wrap(x):
        return b.apply(params, sub_keys[4], x)

    assert_equivariant(wrap, rng_key=sub_keys[5], args_in=[x], irreps_in=[irreps], irreps_out=[irreps])
