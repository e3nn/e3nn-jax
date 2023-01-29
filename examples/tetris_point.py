import time

import flax
import jax
import jax.numpy as jnp
import optax

import e3nn_jax as e3nn
from e3nn_jax.experimental.point_convolution import MessagePassingConvolutionFlax


def tetris():
    pos = jnp.array(
        [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
        ],
        dtype=jnp.float32,
    )

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = jnp.array(
        [
            [+1, 1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
            [-1, 1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
            [0, 0, 1, 0, 0, 0, 0, 0],  # square
            [0, 0, 0, 1, 0, 0, 0, 0],  # line
            [0, 0, 0, 0, 1, 0, 0, 0],  # corner
            [0, 0, 0, 0, 0, 1, 0, 0],  # L
            [0, 0, 0, 0, 0, 0, 1, 0],  # T
            [0, 0, 0, 0, 0, 0, 0, 1],  # zigzag
        ],
        dtype=jnp.float32,
    )

    pos = pos.reshape((8 * 4, 3))
    batch = jnp.arange(8 * 4) // 4

    return pos, labels, batch


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, pos, edge_src, edge_dst):
        pos = e3nn.IrrepsArray("1o", pos)
        node_feat = e3nn.IrrepsArray.ones("0e", (pos.shape[0],), pos.dtype)

        kw = dict(
            radial_basis=lambda r: jnp.ones_like(r)[:, None],
            mlp_neurons=[32, 32],
            avg_num_neighbors=1.5,
            sh_lmax=2,
            num_radial_basis=2,
        )

        for _ in range(4):
            node_feat = MessagePassingConvolutionFlax("32x0e + 32x0o + 16x0e + 8x1e + 8x1o", **kw)(
                pos, node_feat, edge_src, edge_dst
            )
            node_feat = e3nn.gate(node_feat)
        node_feat = MessagePassingConvolutionFlax("0o + 7x0e", **kw)(pos, node_feat, edge_src, edge_dst)

        return node_feat.array


def train(steps=2000):
    model = Model()
    opt = optax.sgd(learning_rate=0.1, momentum=0.9)

    def loss_pred(params, pos, edge_src, edge_dst, labels, batch):
        pred = model.apply(params, pos, edge_src, edge_dst)
        pred = e3nn.scatter_sum(pred, dst=batch, output_size=8)  # [batch, 1 + 7]
        loss_odd = jnp.log(1 + jnp.exp(-labels[:, 0] * pred[:, 0]))
        loss_even = jnp.mean(-labels[:, 1:] * jax.nn.log_softmax(pred[:, 1:]), axis=1)
        loss = jnp.mean(loss_odd + loss_even)
        return loss, pred

    @jax.jit
    def update(params, opt_state, pos, edge_src, edge_dst, labels, batch):
        grad_fn = jax.value_and_grad(loss_pred, has_aux=True)
        (_, pred), grads = grad_fn(params, pos, edge_src, edge_dst, labels, batch)
        accuracy_odd = jnp.sign(jnp.round(pred[:, 0])) == labels[:, 0]
        accuracy_even = jnp.argmax(pred[:, 1:], axis=1) == jnp.argmax(labels[:, 1:], axis=1)
        accuracy = (jnp.mean(accuracy_odd) + jnp.mean(accuracy_even)) / 2
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy, pred

    pos, labels, batch = tetris()
    edge_src, edge_dst = e3nn.radius_graph(pos, 1.1, batch=batch)

    params = model.init(jax.random.PRNGKey(2), pos, edge_src, edge_dst)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...")
    _, _, _, pred = update(params, opt_state, pos, edge_src, edge_dst, labels, batch)
    pred.block_until_ready()

    print(f"It took {time.perf_counter() - wall:.1f}s to compile jit.")

    wall = time.perf_counter()
    for it in range(1, steps + 1):
        params, opt_state, accuracy, pred = update(params, opt_state, pos, edge_src, edge_dst, labels, batch)

        print(f"[{it}] accuracy = {100 * accuracy:.0f}%")

        if accuracy == 1:
            total = time.perf_counter() - wall
            print(f"100% accuracy has been reach in {total:.1f}s after {it} iterations ({1000 * total/it:.1f}ms/it).")
            break

    jnp.set_printoptions(precision=2, suppress=True)
    print(pred)


if __name__ == "__main__":
    train()
