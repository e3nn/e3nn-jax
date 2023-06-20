import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.experimental.transformer import Transformer
from e3nn_jax.utils import assert_equivariant


def test_transformer(keys):
    @hk.without_apply_rng
    @hk.transform
    def model(pos, src, dst, node_feat):
        edge_distance = e3nn.norm(pos[dst] - pos[src]).array[..., 0]
        edge_weight_cutoff = e3nn.sus(3.0 * (2.0 - edge_distance))
        edge_attr = e3nn.concatenate(
            [
                e3nn.soft_one_hot_linspace(
                    edge_distance,
                    start=0.0,
                    end=2.0,
                    number=5,
                    basis="smooth_finite",
                    cutoff=True,
                ),
                e3nn.spherical_harmonics("1e + 2e", pos[dst] - pos[src], True),
            ]
        )
        return Transformer(
            "0e + 2x1e + 2e",
            list_neurons=[32, 32],
            act=jax.nn.relu,
            num_heads=2,
        )(src, dst, edge_weight_cutoff, edge_attr, node_feat)

    apply = jax.jit(model.apply)

    pos = e3nn.IrrepsArray(
        "1e",
        jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    src, dst = e3nn.radius_graph(pos, 2.0)
    node_feat = e3nn.normal("2x0e + 2x1e + 2x2e", next(keys), (pos.shape[0],))

    w = model.init(next(keys), pos, src, dst, node_feat)
    apply(w, pos, src, dst, node_feat)

    assert_equivariant(
        lambda pos, node_feat: apply(w, pos, src, dst, node_feat),
        jax.random.PRNGKey(0),
        pos,
        node_feat,
        atol=1e-4,
    )
