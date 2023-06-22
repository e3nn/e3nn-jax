from typing import Callable, Sequence, Tuple

import flax
import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

_docstring_class = r"""Message passing convolution

Args:
    target_irreps (e3nn.Irreps): irreps of the output
    radial_basis (Callable[[jnp.ndarray], jnp.ndarray]): radial basis functions
    avg_num_neighbors (float): average number of neighbors
    sh_lmax (int): maximum spherical harmonics degree
    num_radial_basis (int): number of radial basis functions
    mlp_neurons (List[int]): number of neurons in each layer of the MLP
    mlp_activation (Callable[[jnp.ndarray], jnp.ndarray]): activation function of the MLP
"""

_docstring_call = r"""Compute the message passing convolution

Args:
    positions (e3nn.IrrepsArray): positions of the nodes
    node_feats (e3nn.IrrepsArray): features of the nodes
    senders (jnp.ndarray): indices of the sender nodes
    receivers (jnp.ndarray): indices of the receiver nodes

Returns:
    e3nn.IrrepsArray: features of the nodes
"""


def radial_basis(r, cutoff, num_radial_basis):
    """Radial basis functions

    This can be used as the `radial_basis` argument of `MessagePassingConvolution`::

        lambda r: radial_basis(r, 6.0, 8)

    Args:
        r (jnp.ndarray): distances
        cutoff (float): cutoff radius
        num_radial_basis (int): number of radial basis functions

    Returns:
        jnp.ndarray: radial basis functions
    """
    # TODO: determine if we need a normalization factor
    r = r / cutoff
    return e3nn.bessel(r, num_radial_basis) * e3nn.soft_envelope(r)[:, None]


def _call(
    self, positions, node_feats, senders, receivers, Linear, MultiLayerPerceptron
):
    if not isinstance(positions, e3nn.IrrepsArray):
        raise TypeError(
            f"positions must be an e3nn.IrrepsArray with shape (n_nodes, 3) and irreps '1o' or '1e'. Got {type(positions)}"
        )
    if not isinstance(node_feats, e3nn.IrrepsArray):
        raise TypeError(
            f"node_feats must be an e3nn.IrrepsArray with shape (n_nodes, irreps). Got {type(node_feats)}"
        )

    assert positions.ndim == 2
    assert node_feats.ndim == 2

    vectors = positions[receivers] - positions[senders]  # [n_edges, 1e or 1o]
    r = e3nn.norm(vectors).array[:, 0]
    edge_attrs = e3nn.concatenate(
        [
            self.radial_basis(r),
            e3nn.spherical_harmonics(list(range(1, self.sh_lmax + 1)), vectors, True),
        ]
    )

    node_feats = Linear(node_feats.irreps, name="linear_up")(node_feats)

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

    mix = MultiLayerPerceptron(
        self.mlp_neurons + (messages.irreps.num_irreps,),
        self.mlp_activation,
        output_activation=False,
    )(
        edge_attrs.filter(keep="0e")
    )  # [n_edges, num_irreps]

    messages = messages * mix  # [n_edges, irreps]

    zeros = e3nn.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
    node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

    node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

    node_feats = Linear(self.target_irreps, name="linear_down")(node_feats)

    return node_feats


class MessagePassingConvolutionHaiku(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        avg_num_neighbors: float,
        sh_lmax: int = 3,
        num_radial_basis: int = 8,
        mlp_neurons: Sequence[int] = (64,),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        name: str = None,
    ):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.radial_basis = radial_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.sh_lmax = sh_lmax
        self.num_radial_basis = num_radial_basis
        self.mlp_neurons = tuple(mlp_neurons)
        self.mlp_activation = mlp_activation

    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [n_edges, 1o or 1e]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]
        return _call(
            self,
            positions,
            node_feats,
            senders,
            receivers,
            e3nn.haiku.Linear,
            e3nn.haiku.MultiLayerPerceptron,
        )


MessagePassingConvolutionHaiku.__doc__ = _docstring_class
MessagePassingConvolutionHaiku.__call__.__doc__ = _docstring_call


class MessagePassingConvolutionFlax(flax.linen.Module):
    target_irreps: e3nn.Irreps
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    avg_num_neighbors: float
    sh_lmax: int = 3
    num_radial_basis: int = 8
    mlp_neurons: Tuple[int, ...] = (64,)
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

    @flax.linen.compact
    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [n_edges, 1o or 1e]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]
        return _call(
            self,
            positions,
            node_feats,
            senders,
            receivers,
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
        )


MessagePassingConvolutionFlax.__doc__ = _docstring_class
MessagePassingConvolutionFlax.__call__.__doc__ = _docstring_call
