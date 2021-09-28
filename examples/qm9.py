import random
import time

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import torch
import torch_geometric as pyg
from e3nn_jax import (Gate, Irreps, index_add, soft_one_hot_linspace,
                      spherical_harmonics)
from e3nn_jax.experimental.point_convolution import Convolution
from torch_geometric.datasets import QM9
from torch_geometric.datasets.qm9 import atomrefs
from tqdm.auto import tqdm


class Sampler():
    def __init__(self, dataset, max_graphs, max_nodes, max_edges, drop_last=True):
        self.num_nodes, self.num_edges = torch.tensor([(a.x.shape[0], a.edge_index.shape[1]) for a in tqdm(dataset)]).T

        self.max_graphs = max_graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.drop_last = drop_last

        assert self.num_nodes.max() <= max_nodes
        assert self.num_edges.max() <= max_edges

    def __iter__(self):
        idx = list(range(len(self.num_nodes)))
        random.shuffle(idx)

        while len(idx) > 0:
            nodes = 0
            edges = 0
            fail = 0
            batch = []

            while fail < 10 and len(idx) > 0 and len(batch) < self.max_graphs:
                v = nodes + self.num_nodes[idx[fail]]
                e = edges + self.num_edges[idx[fail]]
                if (v <= self.max_nodes) and (e <= self.max_edges):
                    batch += [idx.pop(fail)]
                    nodes = v
                    edges = e
                else:
                    fail += 1

            if len(idx) == 0 and self.drop_last:
                break

            yield batch


def dummy_fill(a, num_graphs, num_nodes, num_edges):
    return dict(
        x=torch.cat([a.x, a.x.new_zeros(num_nodes - a.x.shape[0], a.x.shape[1])]),
        pos=torch.cat([a.pos, a.pos.new_zeros(num_nodes - a.pos.shape[0], a.pos.shape[1])]),

        edge_attr=torch.cat([
            a.edge_attr,
            a.edge_attr.new_zeros(num_edges - a.edge_attr.shape[0], a.edge_attr.shape[1])
        ]),

        y=torch.cat([a.y, a.y.new_zeros(num_graphs - a.y.shape[0], a.y.shape[1])]),

        batch=torch.cat([a.batch, (num_graphs - 1) * a.batch.new_ones(num_nodes - a.batch.shape[0])]),
        edge_index=torch.cat([
            a.edge_index,
            (num_nodes - 1) * a.edge_index.new_ones(2, num_edges - a.edge_index.shape[1])
        ], dim=1)
    )


@hk.transform
def f(a):
    irreps_node_attr = Irreps('5x0e')
    node_attr = a['x'][:, :5] * 5**0.5
    pos = a['pos']
    edge_src, edge_dst = a['edge_index']

    irreps_sh = Irreps("0e + 1o + 2e")
    edge_attr = irreps_sh.as_list(spherical_harmonics(
        irreps_sh, pos[edge_dst] - pos[edge_src], True, normalization='component'
    ))

    edge_scalars = soft_one_hot_linspace(
        jnp.linalg.norm(pos[edge_dst] - pos[edge_src], axis=1),
        start=0.0,
        end=1.9,  # max 1.81 in QM9 from pyg,
        number=5,
        basis='smooth_finite',
        cutoff=True,
    ) * 5**0.5

    gate = Gate('256x0e + 256x0o', [jax.nn.gelu, jnp.tanh], '48x0e', [jax.nn.sigmoid], '16x1e + 16x1o + 8x2e + 8x2o')
    g = jax.vmap(gate)

    kw = dict(
        irreps_node_attr=irreps_node_attr,
        irreps_edge_attr=irreps_sh,
        fc_neurons=[64],
        num_neighbors=1.9,
    )

    x = node_attr

    x = g(
        Convolution(
            irreps_node_input=irreps_node_attr,
            irreps_node_output=gate.irreps_in,
            **kw
        )(x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars)
    )

    for _ in range(3):
        x = g(
            Convolution(
                irreps_node_input=gate.irreps_out,
                irreps_node_output=gate.irreps_in,
                **kw
            )(x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars)
        )

    irreps_out = Irreps('4x0e')
    x = Convolution(
        irreps_node_input=gate.irreps_out,
        irreps_node_output=irreps_out,
        **kw
    )(x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars)

    out = irreps_out.as_tensor(x)

    M = jnp.array([atomrefs[i] for i in range(7, 11)]).T

    out = node_attr @ M + out

    return index_add(a['batch'], out, a['y'].shape[0])


def main():
    dataset = QM9('~/qm9')

    num_graphs = 64
    num_nodes = 512
    num_edges = 1024
    learning_rate = 1e-2
    momentum = 0.9

    sampler = Sampler(dataset, num_graphs - 1, num_nodes - 1, num_edges)

    print(f"nodes: min={sampler.num_nodes.min()} med={sampler.num_nodes.median()} max={sampler.num_nodes.max()} tot={sampler.num_nodes.sum()}")
    print(f"edges: min={sampler.num_edges.min()} med={sampler.num_edges.median()} max={sampler.num_edges.max()} tot={sampler.num_edges.sum()}")

    loader = pyg.loader.DataLoader(dataset, batch_sampler=sampler)
    def batch_gen():
        for a in loader:
            a = dummy_fill(a, num_graphs, num_nodes, num_edges)
            a = jax.tree_map(lambda x: jnp.array(x), a)
            yield a

    ##############

    opt = optax.sgd(learning_rate, momentum)

    def loss_pred(params, a):
        pred = f.apply(params, None, a)
        pred = pred.at[-1].set(0.0)  # the last graph is a dummy graph!
        loss = jnp.mean(jnp.sum((pred - a['y'][:, 7:11])**2, axis=1))
        return loss, pred

    def update(params, opt_state, a):
        grad_fn = jax.value_and_grad(loss_pred, has_aux=True)
        (loss, pred), grads = grad_fn(params, a)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    @jax.jit
    def n_updates(n, params, opt_state, a):
        def f(i, x):
            params, opt_state = x
            params, opt_state, _loss, _pred = update(params, opt_state, a)
            return (params, opt_state)

        params, opt_state = jax.lax.fori_loop(0, n - 1, f, (params, opt_state))
        params, opt_state, loss, pred = update(params, opt_state, a)
        return params, opt_state, loss, pred

    key = jax.random.PRNGKey(0)
    params = f.init(key, next(batch_gen()))
    opt_state = opt.init(params)

    for a in batch_gen():
        wall = time.perf_counter()
        params, opt_state, loss, pred = n_updates(100, params, opt_state, a)
        total = time.perf_counter() - wall
        print(f"{10 * total:.3f}ms L={loss:.3f}")


if __name__ == "__main__":
    main()
