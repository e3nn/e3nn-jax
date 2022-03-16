import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import Irreps, radius_graph, spherical_harmonics
from e3nn_jax.experimental.point_convolution import Convolution


def test_transformer(keys):
    irreps_node_input = Irreps("2x0e + 2x1e + 2x2e")
    irreps_node_output = Irreps("0e + 2x1e + 2e")
    irreps_edge_attr = Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(pos, features):
        src, dst = radius_graph(pos, 2.0, size=10)  # TODO check what happens with -1 indices
        edge_attr = spherical_harmonics(irreps_edge_attr, pos[dst] - pos[src], True)

        return Convolution(
            irreps_node_input=irreps_node_input,
            irreps_node_attr=None,
            irreps_edge_attr=irreps_edge_attr,
            irreps_node_output=irreps_node_output,
            fc_neurons=[],
            num_neighbors=2.0,
        )(features, src, dst, edge_attr)

    f = jax.jit(c.apply)

    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    x = irreps_node_input.randn(next(keys), (pos.shape[0], -1))

    w = c.init(next(keys), pos, x)
    f(w, pos, x)

    # TODO test equivariance
