import haiku as hk

from e3nn_jax import Dropout, Irreps, IrrepsArray
from e3nn_jax.util.test import assert_equivariant


def test_dropout(keys):
    irreps = Irreps("10x1e + 10x0e")

    @hk.without_apply_rng
    @hk.transform
    def b(rng, x, is_training=True):
        m = Dropout(p=0.75)
        return m(rng, x, is_training)

    x = IrrepsArray.from_array(irreps, irreps.randn(next(keys), (-1,)))
    params = b.init(next(keys), next(keys), x)

    y = b.apply(params, next(keys), x, is_training=False)
    assert (y.array == x.array).all()

    y = b.apply(params, next(keys), x)
    assert ((y.array == (x.array / 0.25)) | (y.array == 0)).all()

    def wrap(x):
        x = IrrepsArray.from_array(irreps, x)
        return b.apply(params, keys[0], x).array

    assert_equivariant(wrap, rng_key=next(keys), args_in=[x.array], irreps_in=[irreps], irreps_out=[irreps])
