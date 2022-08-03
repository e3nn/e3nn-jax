import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import Irreps, IrrepsArray, radius_graph, spherical_harmonics
from e3nn_jax.experimental.point_convolution import Convolution


def test_point_convolution(keys):
    @hk.without_apply_rng
    @hk.transform
    def model(pos, src, dst, x):
        edge_attr = spherical_harmonics("0e + 1e + 2e", pos[dst] - pos[src], True)

        return Convolution(
            "8x0e + 8x0o + 5e",
            fc_neurons=[],
            num_neighbors=2.0,
        )(x, src, dst, edge_attr)

    apply = jax.jit(model.apply)

    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    x = IrrepsArray.randn("16x0e + 1o", next(keys), (pos.shape[0],))
    src, dst = radius_graph(pos, 2.0)

    w = model.init(next(keys), pos, src, dst, x)
    out = apply(w, pos, src, dst, x)

    assert out.shape[:-1] == x.shape[:-1]
    assert out.irreps == Irreps("8x0e + 8x0o + 5e")
    assert out.list[2] is None

    # TODO test equivariance
