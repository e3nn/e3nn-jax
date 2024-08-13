import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalTensorProduct


class SlowLinear:
    r"""Inefficient implementation of FunctionalLinear relying on TensorProduct."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
    ):
        super().__init__()

        irreps_in = e3nn.Irreps(irreps_in)
        irreps_out = e3nn.Irreps(irreps_out)

        instr = [
            (i_in, 0, i_out, "uvw", True, 1.0)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        self.tp = FunctionalTensorProduct(
            irreps_in,
            "0e",
            irreps_out,
            instr,
        )

        self.output_mask = self.tp.output_mask
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def __call__(self, ws, x):
        ones = e3nn.IrrepsArray("0e", jnp.array([1.0]))
        return self.tp.left_right(ws, x, ones)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_like_tp(keys, irreps_in, irreps_out):
    """Test that Linear gives the same results as the corresponding TensorProduct."""
    m = e3nn.FunctionalLinear(irreps_in, irreps_out)
    m_tp = SlowLinear(irreps_in, irreps_out)

    ws = [jax.random.normal(next(keys), i.path_shape) for i in m.instructions]
    ws_tp = [w[:, None, :] for w in ws]
    x = e3nn.normal(m.irreps_in, next(keys), ())
    assert jnp.allclose(m(ws, x).array, m_tp(ws_tp, x).array)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_matrix(keys, irreps_in, irreps_out):
    m = e3nn.FunctionalLinear(irreps_in, irreps_out)

    ws = [jax.random.normal(next(keys), i.path_shape) for i in m.instructions]
    x = e3nn.normal(m.irreps_in, next(keys), ())

    A = m.matrix(ws)
    y1 = x.array @ A
    y2 = m(ws, x).array
    assert jnp.allclose(y1, y2)


def test_normalization_1(keys):
    irreps_in = "10x0e + 20x0e"
    irreps_out = "0e"

    @hk.without_apply_rng
    @hk.transform
    def linear(x):
        return e3nn.haiku.Linear(irreps_out)(x)

    x = e3nn.normal(irreps_in, next(keys), (1024,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3


def test_normalization_2(keys):
    irreps_in = "10x0e + 20x0e"
    irreps_out = "0e"

    @hk.without_apply_rng
    @hk.transform
    def linear(x):
        return e3nn.haiku.Linear(irreps_out, 5)(x)

    x = e3nn.normal(irreps_in, next(keys), (1024, 9))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3


def test_normalization_3(keys):
    irreps_in = "10x0e + 20x0e"
    irreps_out = "0e"

    @hk.without_apply_rng
    @hk.transform
    def linear(x):
        w = hk.get_parameter("w", (32,), init=hk.initializers.Constant(1.0))
        return e3nn.haiku.Linear(irreps_out, 5)(w, x)

    x = e3nn.normal(irreps_in, next(keys), (1024, 9))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize("initializer", ["uniform", "custom"])
def test_linear_vanilla_custom_initializer(keys, irreps_in, irreps_out, initializer):
    if initializer == "uniform":

        def parameter_initializer() -> hk.initializers.Initializer:
            return hk.initializers.UniformScaling(0.1)

    elif initializer == "custom":

        def parameter_initializer() -> hk.initializers.Initializer:
            def custom_initializer(shape, dtype):
                rng = hk.next_rng_key()
                return 5 + jax.random.normal(rng, shape, dtype=jnp.float32) * 0.1

            return custom_initializer

    @hk.without_apply_rng
    @hk.transform
    def linear(x):
        return e3nn.haiku.Linear(
            irreps_out, parameter_initializer=parameter_initializer
        )(x)

    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_vanilla_custom_instructions(keys, irreps_in, irreps_out):
    irreps_in = e3nn.Irreps(irreps_in).simplify()
    irreps_out = e3nn.Irreps(irreps_out).simplify()

    # Keep random instructions.
    instructions = [
        (i_in, i_out)
        for i_in, (_, ir_in) in enumerate(irreps_in)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_in == ir_out and jax.random.bernoulli(next(keys))
    ]

    @hk.without_apply_rng
    @hk.transform
    def linear(x):
        return e3nn.haiku.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            instructions=instructions,
            simplify_irreps_internally=False,
        )(x)

    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)

    # All output irreps in instructions should be non-zero.
    # The other output irreps should be zero, as they are inaccessible.
    output_indices = set(i_out for _, i_out in instructions)
    for i_out, irreps_slice in enumerate(y.irreps.slices()):
        if i_out in output_indices:
            assert jnp.all(y.array[..., irreps_slice] != 0.0)
        else:
            assert jnp.all(y.array[..., irreps_slice] == 0.0)

    assert y.shape == (128, y.irreps.dim)
