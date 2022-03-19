import time

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from e3nn_jax import (gate, Irreps, IrrepsData, index_add, radius_graph,
                      spherical_harmonics)
from e3nn_jax.experimental.point_convolution import Convolution


def tetris():
    pos = jnp.array([
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ], dtype=jnp.float32)

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = jnp.array([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=jnp.float32)

    pos = pos.reshape((8 * 4, 3))
    batch = jnp.arange(8 * 4) // 4

    return pos, labels, batch


def model(x, edge_src, edge_dst, edge_attr):
    x = IrrepsData.from_contiguous("0e", x)
    edge_attr = IrrepsData.from_list("0e + 1o + 2e", edge_attr, (edge_src.shape[0],))

    kw = dict(
        irreps_node_attr=Irreps('0e'),
        fc_neurons=None,
        num_neighbors=1.5,
    )

    for _ in range(4):
        x = Convolution(
            irreps_node_output='32x0e + 32x0o + 16x0e + 8x1e + 8x1o',
            **kw
        )(x, edge_src, edge_dst, edge_attr)
        x = jax.vmap(gate, (0, None), 0)(x, [jax.nn.gelu, jnp.tanh, jax.nn.sigmoid])

    x = Convolution(
        irreps_node_output='0o + 6x0e',
        **kw
    )(x, edge_src, edge_dst, edge_attr)

    return x.contiguous


def main():
    @hk.transform
    def f(input):
        node_input, edge_src, edge_dst, edge_attr = input
        return model(node_input, edge_src, edge_dst, edge_attr)

    opt = optax.sgd(learning_rate=0.1, momentum=0.9)

    def loss_pred(params, input, labels, batch):
        pred = f.apply(params, None, input)
        pred = index_add(batch, pred, 8)
        loss = jnp.mean((pred - labels)**2)
        return loss, pred

    def update(params, opt_state, input, labels, batch):
        grad_fn = jax.value_and_grad(loss_pred, has_aux=True)
        (loss, pred), grads = grad_fn(params, input, labels, batch)
        accuracy = jnp.mean(jnp.all(jnp.round(pred) == labels, axis=1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy, pred

    @jax.jit
    def update_n(n, params, opt_state, input, labels, batch):
        def body(_i, state):
            return update(*state, input, labels, batch)[:2]

        params, opt_state = jax.lax.fori_loop(0, n - 1, body, (params, opt_state))
        return update(params, opt_state, input, labels, batch)

    jnp.set_printoptions(precision=2, suppress=True)

    pos, labels, batch = tetris()
    edge_src, edge_dst = radius_graph(pos, 1.1, batch)
    irreps_sh = Irreps("0e + 1o + 2e")
    edge_attr = spherical_harmonics(irreps_sh, pos[edge_dst] - pos[edge_src], True, normalization='component').list
    node_input = jnp.ones((pos.shape[0], 1))
    input = (node_input, edge_src, edge_dst, edge_attr)

    params = f.init(jax.random.PRNGKey(3), input)
    opt_state = opt.init(params)

    jnp.set_printoptions(precision=2, suppress=True)

    # compile jit
    wall = time.perf_counter()
    print("compiling...")
    _, _, _, _, pred = update_n(2, params, opt_state, input, labels, batch)
    print(pred)

    print(f"It took {time.perf_counter() - wall:.1f}s to compile jit.")

    wall = time.perf_counter()
    it = 0
    for _ in range(2000):
        params, opt_state, loss, accuracy, pred = update_n(100, params, opt_state, input, labels, batch)
        it += 100
        if accuracy == 1:
            break

    total = time.perf_counter() - wall
    print(f"100% accuracy has been reach in {total:.1f}s after {it} iterations ({1000 * total/it:.1f}ms/it).")

    print(f"accuracy = {100 * accuracy:.0f}%")

    print(pred)


if __name__ == '__main__':
    main()
