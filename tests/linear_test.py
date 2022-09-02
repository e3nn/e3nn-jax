import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

import pytest


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

        self.tp = e3nn.FunctionalTensorProduct(
            irreps_in,
            "0e",
            irreps_out,
            instr,
        )

        self.output_mask = self.tp.output_mask
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def __call__(self, ws, x):
        ones = e3nn.IrrepsArray.ones("0e", ())
        return self.tp.left_right(ws, x, ones)


@pytest.mark.parametrize("irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"])
@pytest.mark.parametrize("irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"])
def test_linear_like_tp(keys, irreps_in, irreps_out):
    """Test that Linear gives the same results as the corresponding TensorProduct."""
    m = e3nn.FunctionalLinear(irreps_in, irreps_out)
    m_tp = SlowLinear(irreps_in, irreps_out)

    ws = [jax.random.normal(next(keys), i.path_shape) for i in m.instructions]
    ws_tp = [w[:, None, :] for w in ws]
    x = e3nn.normal(m.irreps_in, next(keys), ())
    assert jnp.allclose(m(ws, x).array, m_tp(ws_tp, x).array)


@pytest.mark.parametrize("irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"])
@pytest.mark.parametrize("irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"])
def test_linear_matrix(keys, irreps_in, irreps_out):
    m = e3nn.FunctionalLinear(irreps_in, irreps_out)

    ws = [jax.random.normal(next(keys), i.path_shape) for i in m.instructions]
    x = e3nn.normal(m.irreps_in, next(keys), ())

    A = m.matrix(ws)
    y1 = x.array @ A
    y2 = m(ws, x).array
    assert jnp.allclose(y1, y2)
