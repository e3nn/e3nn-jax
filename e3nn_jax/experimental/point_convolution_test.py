import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax.experimental.point_convolution import (
    MessagePassingConvolutionHaiku,
    radial_basis,
)
from e3nn_jax.utils import assert_equivariant, assert_output_dtype_matches_input_dtype


def test_point_convolution(keys):
    jax.config.update("jax_enable_x64", True)
    cutoff = 2.0

    @hk.without_apply_rng
    @hk.transform
    def model(
        positions: e3nn.IrrepsArray,
        features: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        return MessagePassingConvolutionHaiku(
            "8x0e + 8x0o + 5e",
            lambda r: radial_basis(r, cutoff, 8),
            avg_num_neighbors=2.0,
            mlp_neurons=[64],
        )(positions, features, senders, receivers).remove_zero_chunks()

    model_init = jax.jit(model.init)
    model_apply = jax.jit(model.apply)

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

    w = model_init(next(keys), pos, feat, src, dst)
    out = model_apply(w, pos, feat, src, dst)

    assert out.shape[:-1] == feat.shape[:-1]
    assert out.irreps == e3nn.Irreps("8x0e")

    assert_equivariant(
        lambda pos, x: model_apply(w, pos, x, src, dst),
        jax.random.PRNGKey(0),
        pos,
        feat,
    )
    assert_output_dtype_matches_input_dtype(model_apply, w, pos, feat, src, dst)
