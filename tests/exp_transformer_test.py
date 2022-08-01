import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import IrrepsArray, radius_graph, spherical_harmonics, sus, soft_one_hot_linspace
from e3nn_jax.experimental.transformer import Transformer


def test_transformer(keys):
    @hk.without_apply_rng
    @hk.transform
    def model(pos, src, dst, node_feat):
        edge_attr = spherical_harmonics("0e + 1e + 2e", pos[dst] - pos[src], True)
        edge_distance = jnp.linalg.norm(pos[dst] - pos[src], axis=-1)
        edge_weight_cutoff = sus(3.0 * (2.0 - edge_distance))
        edge_scalar_attr = soft_one_hot_linspace(
            edge_distance, start=0.0, end=2.0, number=5, basis="smooth_finite", cutoff=True
        )

        return Transformer(
            "0e + 2x1e + 2e",
            list_neurons=[32, 32],
            act=jax.nn.relu,
            num_heads=2,
        )(src, dst, edge_scalar_attr, edge_weight_cutoff, edge_attr, node_feat)

    apply = jax.jit(model.apply)

    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    src, dst = radius_graph(pos, 2.0)
    node_feat = IrrepsArray.randn("2x0e + 2x1e + 2x2e", next(keys), (pos.shape[0],))

    w = model.init(next(keys), pos, src, dst, node_feat)
    apply(w, pos, src, dst, node_feat)

    # TODO test equivariance
