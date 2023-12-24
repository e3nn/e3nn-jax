import haiku as hk

import e3nn_jax as e3nn
from e3nn_jax.utils import assert_equivariant


def test_dropout(keys):
    irreps = e3nn.Irreps("10x1e + 10x0e")

    @hk.without_apply_rng
    @hk.transform
    def b(rng, x, is_training=True):
        m = e3nn.haiku.Dropout(p=0.75)
        return m(rng, x, is_training)

    x = e3nn.normal(irreps, next(keys), ())
    params = b.init(next(keys), next(keys), x)

    y = b.apply(params, next(keys), x, is_training=False)
    assert (y.array == x.array).all()

    y = b.apply(params, next(keys), x)
    assert ((y.array == (x.array / 0.25)) | (y.array == 0)).all()

    def wrap(x):
        return b.apply(params, keys[0], x)

    assert_equivariant(wrap, next(keys), x)
