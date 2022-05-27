# TODO update this code to the new API (IrrepsData)
import argparse
import random
import time
from itertools import count

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.multiprocessing
import torch_geometric as pyg
import wandb
from e3nn_jax import Gate, Irreps, IrrepsData, index_add, soft_one_hot_linspace, spherical_harmonics
from e3nn_jax.experimental.point_convolution import Convolution
from torch_geometric.datasets import QM9
from torch_geometric.datasets.qm9 import atomrefs
from tqdm.auto import tqdm


class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0
        self.running = False

    def start(self):
        assert not self.running
        self.t -= time.perf_counter()
        self.running = True

    def stop(self, n=1):
        assert self.running
        self.t += time.perf_counter()
        self.n += n
        self.running = False

    def reset(self):
        self.t = 0.0
        self.n = 0
        self.running = False

    def __call__(self):
        return self.t / self.n

    def __repr__(self):
        t = self.t / self.n
        if t < 1:
            return f"{1000 * t:.1f}ms"
        return f"{t:.2f}s"


class Sampler:
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

            while fail < 10 and fail < len(idx) and len(batch) < self.max_graphs:
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
        edge_attr=torch.cat([a.edge_attr, a.edge_attr.new_zeros(num_edges - a.edge_attr.shape[0], a.edge_attr.shape[1])]),
        num_graphs=torch.tensor(a.y.shape[0]),
        y=torch.cat([a.y, a.y.new_zeros(num_graphs - a.y.shape[0], a.y.shape[1])]),
        batch=torch.cat([a.batch, (num_graphs - 1) * a.batch.new_ones(num_nodes - a.batch.shape[0])]),
        edge_index=torch.cat(
            [a.edge_index, (num_nodes - 1) * a.edge_index.new_ones(2, num_edges - a.edge_index.shape[1])], dim=1
        ),
    )


def create_model(config):
    @hk.transform
    def f(a):
        irreps_node_attr = Irreps("5x0e")
        node_attr = a["x"][:, :5] * 5 ** 0.5
        pos = a["pos"]
        edge_src, edge_dst = a["edge_index"]

        irreps_sh = Irreps.spherical_harmonics(config["shlmax"])
        edge_attr = IrrepsData.from_contiguous(
            irreps_sh, spherical_harmonics(irreps_sh, pos[edge_dst] - pos[edge_src], True, normalization="component")
        ).list

        edge_scalars = (
            soft_one_hot_linspace(
                jnp.linalg.norm(pos[edge_dst] - pos[edge_src], axis=1),
                start=0.0,
                end=1.9,  # max 1.81 in QM9 from pyg,
                number=config["num_basis"],
                basis="smooth_finite",
                cutoff=True,
            )
            * config["num_basis"] ** 0.5
            * 0.95
        )

        mul0 = config["mul0"]
        mul1 = config["mul1"]
        mul2 = config["mul2"]
        irreps_gated = Irreps(f"{mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o").simplify()

        gate = Gate(
            f"{mul0}x0e + {mul0}x0o", [jax.nn.gelu, jnp.tanh], f"{irreps_gated.num_irreps}x0e", [jax.nn.sigmoid], irreps_gated
        )
        g = jax.vmap(gate)

        kw = dict(
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_sh,
            fc_neurons=[config["radial_num_neurons"]] * config["radial_num_layers"],
            num_neighbors=2.1,
        )

        x = node_attr

        # def stat(text, z):
        #     print(f"{text} = {jax.tree_map(lambda x: float(jnp.mean(jnp.mean(x**2, axis=1))), z)}")

        # stat('input', x)
        # stat('edge_attr', edge_attr)
        # stat('node_attr', node_attr)
        # stat('edge_scalars', edge_scalars)

        x = Convolution(irreps_node_input=irreps_node_attr, irreps_node_output=gate.irreps_in, **kw)(
            x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars
        )

        # print()
        # stat('c(x)', x)

        x = g(x)

        # stat('g(c(x))', x)

        for _ in range(config["num_layers"]):
            x = g(
                Convolution(irreps_node_input=gate.irreps_out, irreps_node_output=gate.irreps_in, **kw)(
                    x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars
                )
            )
            # stat('x', x)

        irreps_out = Irreps("4x0e")
        x = Convolution(irreps_node_input=gate.irreps_out, irreps_node_output=irreps_out, **kw)(
            x, edge_src, edge_dst, edge_attr, node_attr=node_attr, edge_scalar_attr=edge_scalars
        )

        # stat('x', x)

        out = x.contiguous

        M = jnp.array([atomrefs[i] for i in range(7, 11)]).T

        out = a["x"][:, :5] @ M + out

        return index_add(a["batch"], out, a["y"].shape[0])

    return f


def execute(config):
    dataset = QM9(config["data_path"])

    sampler = Sampler(dataset, config["num_graphs"] - 1, config["num_nodes"] - 1, config["num_edges"])

    print(
        f"nodes: min={sampler.num_nodes.min()} med={sampler.num_nodes.median()} max={sampler.num_nodes.max()} "
        f"tot={sampler.num_nodes.sum()}"
    )
    print(
        f"edges: min={sampler.num_edges.min()} med={sampler.num_edges.median()} max={sampler.num_edges.max()} "
        f"tot={sampler.num_edges.sum()}"
    )

    loader = pyg.loader.DataLoader(dataset, batch_sampler=sampler, num_workers=2)

    def batch_gen():
        for a in loader:
            a = dummy_fill(a, config["num_graphs"], config["num_nodes"], config["num_edges"])
            a = jax.tree_map(lambda x: np.array(x), a)
            yield a

    ##############
    f = create_model(config)
    opt = optax.sgd(config["lr"], config["momentum"])

    def loss_pred(params, a):
        pred = f.apply(params, None, a)
        pred = pred.at[-1].set(0.0)  # the last graph is a dummy graph!
        loss = jnp.sum(jnp.abs(pred - a["y"][:, 7:11])) / a["num_graphs"]
        return loss, pred

    @jax.jit
    def update(params, opt_state, a):
        grad_fn = jax.value_and_grad(loss_pred, has_aux=True)
        (loss, pred), grads = grad_fn(params, a)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    print("init model...", flush=True)
    key = jax.random.PRNGKey(config["seed"])
    params = f.init(key, next(batch_gen()))
    opt_state = opt.init(params)

    print("compiling...", flush=True)
    update(params, opt_state, next(batch_gen()))

    # jax.profiler.start_trace("/tmp/tensorboard")

    wall = time.perf_counter()
    t_update = Timer()
    t_all = Timer()
    i = 0
    mae = []

    t_all.start()
    for epoch in count():
        for a in batch_gen():
            t_update.start()
            params, opt_state, loss, pred = update(params, opt_state, a)
            loss, pred = jax.tree_map(np.array, (loss, pred))
            t_update.stop()

            mae += [np.abs(pred - a["y"][:, 7:11])[: a["num_graphs"]]]

            if i % 100 == 0:
                mae = mae[-5000:]
                e = 1000 * np.mean(np.concatenate(mae, axis=0), axis=0)

                t_all.stop(100)
                print((f"E={epoch} i={i} " f"step={t_update}/{t_all} " f"mae={list(np.round(e, 2))}meV"), flush=True)

                status = {
                    "epoch": epoch,
                    "iteration": i,
                    "_runtime": time.perf_counter() - wall,
                    "dt1": t_update(),
                    "dt2": t_all(),
                    "train": {
                        "mae_total": np.sum(e),
                        "mae_7": e[7 - 7],
                        "mae_8": e[8 - 7],
                        "mae_9": e[9 - 7],
                        "mae_10": e[10 - 7],
                    },
                }
                wandb.log(status)

                t_update.reset()
                t_all.reset()
                t_all.start()

            i += 1
    # jax.profiler.stop_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mul0", type=int, default=128)
    parser.add_argument("--mul1", type=int, default=128)
    parser.add_argument("--mul2", type=int, default=128)
    parser.add_argument("--shlmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_basis", type=int, default=10)

    parser.add_argument("--radial_num_neurons", type=int, default=64)
    parser.add_argument("--radial_num_layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_graphs", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=512)
    parser.add_argument("--num_edges", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runtime", type=int, default=(3 * 24 - 1) * 3600)
    parser.add_argument("--data_path", type=str, default="~/qm9")

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")

    # wandb.login()
    wandb.init(project="QM9 jax", config=args.__dict__)
    config = dict(wandb.config)
    # config = args.__dict__
    print(config)
    execute(config)


if __name__ == "__main__":
    main()
