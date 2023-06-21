import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp


def test_normalization():
    x = e3nn.normal("0e + 1e + 2e", jax.random.PRNGKey(5), (16, 1024))

    @hk.without_apply_rng
    @hk.transform
    def model(x):
        return e3nn.haiku.SymmetricTensorProduct((3,))(x)

    w = jax.jit(model.init)(jax.random.PRNGKey(1), x)
    y = jax.jit(model.apply)(w, x)

    for (mul, ir), e in zip(y.irreps, y.chunks):
        assert 0.5 < float(jnp.mean(e**2)) < 6.5
