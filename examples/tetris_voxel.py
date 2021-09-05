import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from e3nn_jax import Gate, Irreps
from e3nn_jax.experimental.voxel_convolution import Convolution
from flax.training import train_state
from tqdm.auto import tqdm


def tetris():
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]

    voxels = np.zeros((8, 8, 8, 8, 1))
    for ps, v in zip(pos, voxels):
        for x, y, z in ps:
            v[3 + x, 3 + y, 3 + z] = 1

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = np.array([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=np.float32)

    return voxels, labels


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        gate = Gate('10x0e + 10x0o', [jax.nn.gelu, jnp.tanh], '10x0e', [jax.nn.sigmoid], '5x1e + 5x1o')

        def g(x):
            y = jax.vmap(gate)(x.reshape(-1, x.shape[-1]))
            return y.reshape(x.shape[:-1] + (-1,))

        kw = dict(irreps_sh=Irreps('0e + 1o'), diameter=1.5, num_radial_basis=2, steps=(1.0, 1.0, 1.0))

        print(x.shape)
        x = g(Convolution(Irreps('0e'), gate.irreps_in, **kw)(x))
        print(x.shape)

        for _ in range(3):
            x = g(Convolution(gate.irreps_out, gate.irreps_in, **kw)(x))
            print(x.shape)

        x = Convolution(gate.irreps_out, Irreps('0o + 6x0e'), **kw)(x)
        print(x.shape)

        x = jnp.sum(x, axis=(1, 2, 3))
        return x


@jax.jit
def apply_model(state, consts, x, y):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        pred = Model().apply({'params': params, 'consts': consts}, x)
        loss = jnp.mean((pred - y)**2)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.all(jnp.round(pred) == y, axis=1))
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def main():
    x, y = tetris()

    learning_rate = 0.1
    momentum = 0.9

    rng = jax.random.PRNGKey(3)

    model = Model()
    params = model.init(rng, x)
    consts = params['consts']

    tx = optax.sgd(learning_rate, momentum)
    st = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx
    )

    for _ in tqdm(range(100)):
        grads, loss, accuracy = apply_model(st, consts, x, y)
        st = update_model(st, grads)
        print(f"loss = {loss:.3f}")

    print(f"accuracy = {100 * accuracy:.0f}%")


if __name__ == '__main__':
    main()
