from typing import Callable, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


class MessagePassingConvolution(hk.Module):
    r"""Message passing convolution

    Args:
        target_irreps (e3nn.Irreps): irreps of the output
        avg_num_neighbors (float): average number of neighbors
        sh_lmax (int): maximum spherical harmonics degree
        num_radial_basis (int): number of radial basis functions
        mlp_neurons (List[int]): number of neurons in each layer of the MLP
        mlp_activation (Callable[[jnp.ndarray], jnp.ndarray]): activation function of the MLP

    Returns:
        e3nn.IrrepsArray: output features of the nodes
    """

    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        *,
        avg_num_neighbors: float,
        sh_lmax: int = 3,
        num_radial_basis: int = 8,
        mlp_neurons: Sequence[int] = (64,),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
    ):
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.avg_num_neighbors = avg_num_neighbors
        self.sh_lmax = sh_lmax
        self.num_radial_basis = num_radial_basis
        self.mlp_neurons = list(mlp_neurons)
        self.mlp_activation = mlp_activation

    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [n_edges, 1o or 1e]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
        cutoff: float,
    ) -> e3nn.IrrepsArray:
        assert positions.ndim == 2
        assert node_feats.ndim == 2

        vectors = positions[receivers] - positions[senders]  # [n_edges, 1e or 1o]
        r = e3nn.norm(vectors).array[:, 0]
        edge_attrs = e3nn.concatenate(
            [
                e3nn.bessel(r, self.num_radial_basis, cutoff) * e3nn.soft_envelope(r, cutoff)[:, None],
                e3nn.spherical_harmonics(list(range(1, self.sh_lmax + 1)), vectors, True),
            ]
        )

        node_feats = e3nn.haiku.Linear(node_feats.irreps, name="linear_up")(node_feats)

        messages = node_feats[senders]

        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    edge_attrs.filter(drop="0e"),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        mix = e3nn.haiku.MultiLayerPerceptron(
            self.mlp_neurons + [messages.irreps.num_irreps],
            self.mlp_activation,
            output_activation=False,
        )(
            edge_attrs.filter(keep="0e")
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

        node_feats = e3nn.haiku.Linear(self.target_irreps, name="linear_down")(node_feats)

        return node_feats
