import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalTensorProduct


class SlowLinear:
    r"""Inefficient implimentation of FunctionalLinear relying on TensorProduct."""

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
    def model(x):
        return e3nn.haiku.Linear(irreps_out)(x)

    x = e3nn.normal(irreps_in, next(keys), (1024,))
    w = model.init(next(keys), x)
    y = model.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3


def test_normalization_2(keys):
    irreps_in = "10x0e + 20x0e"
    irreps_out = "0e"

    @hk.without_apply_rng
    @hk.transform
    def model(x):
        return e3nn.haiku.Linear(irreps_out, 5)(x)

    x = e3nn.normal(irreps_in, next(keys), (1024, 9))
    w = model.init(next(keys), x)
    y = model.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3


def test_normalization_3(keys):
    irreps_in = "10x0e + 20x0e"
    irreps_out = "0e"

    @hk.without_apply_rng
    @hk.transform
    def model(x):
        w = hk.get_parameter("w", (32,), init=hk.initializers.Constant(1.0))
        return e3nn.haiku.Linear(irreps_out, 5)(w, x)

    x = e3nn.normal(irreps_in, next(keys), (1024, 9))
    w = model.init(next(keys), x)
    y = model.apply(w, x)

    assert np.exp(np.abs(np.log(np.mean(y.array**2)))) < 1.3
