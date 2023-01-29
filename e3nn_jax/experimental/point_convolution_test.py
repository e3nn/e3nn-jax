import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax.experimental.point_convolution import MessagePassingConvolution
from e3nn_jax.util import assert_equivariant


def test_point_convolution(keys):
    @hk.without_apply_rng
    @hk.transform
    def model(
        positions: e3nn.IrrepsArray,
        features: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        cutoff: float,
    ) -> e3nn.IrrepsArray:
        return MessagePassingConvolution(
            "8x0e + 8x0o + 5e",
            avg_num_neighbors=2.0,
            mlp_neurons=[64],
        )(positions, features, senders, receivers, cutoff)

    model_init = jax.jit(model.init)
    model_apply = jax.jit(model.apply)

    cutoff = 2.0
    pos = e3nn.IrrepsArray(
        "1o",
        jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )
    feat = e3nn.normal("16x0e + 1o", next(keys), (pos.shape[0],))
    src, dst = e3nn.radius_graph(pos, cutoff)

    w = model_init(next(keys), pos, feat, src, dst, cutoff)
    out = model_apply(w, pos, feat, src, dst, cutoff)

    assert out.shape[:-1] == feat.shape[:-1]
    assert out.irreps == e3nn.Irreps("8x0e + 8x0o + 5e")
    assert out.list[2] is None

    assert_equivariant(
        lambda pos, x: model_apply(w, pos, x, src, dst, cutoff),
        jax.random.PRNGKey(0),
        args_in=[pos, feat],
    )
