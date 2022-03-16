import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import Irreps, radius_graph, spherical_harmonics, sus
from e3nn_jax.experimental.transformer import Transformer


def test_transformer(keys):
    irreps_node_input = Irreps("2x0e + 2x1e + 2x2e")
    irreps_node_output = Irreps("0e + 2x1e + 2e")
    irreps_edge_attr = Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(pos, features):
        src, dst = radius_graph(pos, 2.0, size=10)  # TODO check what happens with -1 indices
        edge_attr = spherical_harmonics(irreps_edge_attr, pos[dst] - pos[src], True)
        edge_weight_cutoff = sus(3.0 * (2.0 - jnp.linalg.norm(pos[dst] - pos[src], axis=-1)))

        return Transformer(
            irreps_node_input=irreps_node_input,
            irreps_node_output=irreps_node_output,
            irreps_edge_attr=irreps_edge_attr,
            list_neurons=[32, 32],
            phi=jax.nn.relu,
            num_heads=2,
        )(src, dst, jnp.ones((src.shape[0], 1)), edge_attr, edge_weight_cutoff, features)

    f = jax.jit(c.apply)

    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    x = irreps_node_input.randn(next(keys), (pos.shape[0], -1))

    w = c.init(next(keys), pos, x)
    f(w, pos, x)

    # TODO test equivariance
