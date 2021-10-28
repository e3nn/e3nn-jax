import haiku as hk

from e3nn_jax import Dropout, Irreps
from e3nn_jax.util.test import assert_equivariant


def test_dropout(keys):
    irreps = Irreps("10x1e + 10x0e")

    @hk.without_apply_rng
    @hk.transform
    def b(rng, x, is_training=True):
        m = Dropout(irreps=irreps, p=0.75)
        return m(rng, x, is_training)

    params = b.init(next(keys), next(keys), irreps.randn(next(keys), (5, 2, -1)))
    x = irreps.randn(next(keys), (5, 2, -1))

    y = b.apply(params, next(keys), x, is_training=False)
    assert (y == x).all()

    y = b.apply(params, next(keys), x)
    assert ((y == (x / 0.25)) | (y == 0)).all()

    def wrap(x):
        return b.apply(params, keys[0], x)

    assert_equivariant(wrap, rng_key=next(keys), args_in=[x], irreps_in=[irreps], irreps_out=[irreps])
