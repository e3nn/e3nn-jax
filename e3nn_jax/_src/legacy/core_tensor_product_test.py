import jax
import jax.numpy as jnp
import pytest

import e3nn_jax as e3nn
from e3nn_jax.legacy import (
    FunctionalFullyConnectedTensorProduct,
    FunctionalTensorProduct,
)


@pytest.mark.parametrize("custom_einsum_jvp", [False, True])
@pytest.mark.parametrize("connection_mode", ["uvw", "uvu", "uvv"])
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm"])
def test_modes(keys, irrep_normalization, jitted, connection_mode, custom_einsum_jvp):
    tp = FunctionalTensorProduct(
        e3nn.Irreps("10x0o + 10x1o + 1x2e"),
        e3nn.Irreps("10x0o + 10x1o + 1x2o"),
        e3nn.Irreps("10x0e + 10x1e + 2x2o"),
        [
            (0, 0, 0, connection_mode, True),
            (1, 1, 1, connection_mode, True),
            (1, 0, 1, connection_mode, True),
            (2, 2, 2, "uvw", True),
            (2, 1, 2, "uvw", True),
        ],
        irrep_normalization=irrep_normalization,
    )

    def f(ws, x1, x2):
        return tp.left_right(
            ws,
            x1,
            x2,
            custom_einsum_jvp=custom_einsum_jvp,
        )

    if jitted:
        f = jax.jit(f)

    g = tp.left_right

    ws = [
        jax.random.normal(next(keys), ins.path_shape)
        for ins in tp.instructions
        if ins.has_weight
    ]
    x1 = e3nn.normal(tp.irreps_in1, next(keys), ())
    x2 = e3nn.normal(tp.irreps_in2, next(keys), ())

    a = f(ws, x1, x2).array
    b = g(ws, x1, x2).array
    assert jnp.allclose(a, b, rtol=1e-4, atol=1e-6), jnp.max(jnp.abs(a - b))


def test_zero_dim(keys):
    tp = FunctionalTensorProduct(
        "0x0e + 1e",
        "0e + 0x1e",
        "0x0e + 1e",
        [
            (0, 0, 0, "uvw", True),
            (1, 1, 0, "uvw", True),
        ],
    )
    w = [jax.random.normal(keys[1], ins.path_shape) for ins in tp.instructions]
    x = e3nn.normal(tp.irreps_in1, keys[2], ())
    y = e3nn.normal(tp.irreps_in2, keys[3], ())

    assert jnp.allclose(
        tp.left_right(w, x, y, fused=True).array,
        tp.left_right(w, x, y, fused=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fused(keys):
    tp = FunctionalTensorProduct(
        "10x0e + 5x1e",
        "0e + 1e + 3x1e",
        "10x0e + 5x1e + 30x1e",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 1, "uvu", True),
            (1, 0, 1, "uvu", True),
            (0, 2, 2, "uvuv", True),
        ],
    )
    w = [jax.random.normal(keys[1], ins.path_shape) for ins in tp.instructions]
    x = e3nn.normal(tp.irreps_in1, keys[2], ())
    y = e3nn.normal(tp.irreps_in2, keys[3], ())

    assert jnp.allclose(
        tp.left_right(w, x, y, fused=True).array,
        tp.left_right(w, x, y, fused=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fused_no_weight(keys):
    tp = FunctionalTensorProduct(
        "10x0e",
        "10x0e",
        "10x0e",
        [
            (0, 0, 0, "uuu", False),
        ],
    )
    w = jnp.ones(0)
    x = e3nn.normal(tp.irreps_in1, keys[2], ())
    y = e3nn.normal(tp.irreps_in2, keys[3], ())

    assert jnp.allclose(
        tp.left_right(w, x, y, fused=True).array,
        tp.left_right(w, x, y, fused=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fused_mix_weight(keys):
    tp = FunctionalTensorProduct(
        "5x0e",
        "5x0e",
        "5x0e",
        [
            (0, 0, 0, "uuu", False),
            (0, 0, 0, "uvw", True),
        ],
    )
    w = jax.random.normal(keys[1], (5**3,))
    x = e3nn.normal(tp.irreps_in1, keys[2], ())
    y = e3nn.normal(tp.irreps_in2, keys[3], ())

    assert jnp.allclose(
        tp.left_right(w, x, y, fused=True).array,
        tp.left_right(w, x, y, fused=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fuse(keys):
    tp = FunctionalFullyConnectedTensorProduct("2x0e+1e", "0e+1e", "1e+0e")

    ws = [
        jax.random.normal(next(keys), ins.path_shape)
        for ins in tp.instructions
        if ins.has_weight
    ]
    wf = jnp.concatenate([w.flatten() for w in ws])
    x1 = e3nn.normal(tp.irreps_in1, next(keys), ())
    x2 = e3nn.normal(tp.irreps_in2, next(keys), ())

    a = tp.left_right(ws, x1, x2, fused=False).array
    b = tp.left_right(wf, x1, x2, fused=True).array
    assert jnp.allclose(a, b, rtol=1e-4, atol=1e-6), (a, b)


@pytest.mark.parametrize("gradient_normalization", ["element", "path", 0.5])
@pytest.mark.parametrize("path_normalization", ["element", "path", 0.5])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm"])
def test_normalization(
    keys, irrep_normalization, path_normalization, gradient_normalization
):
    tp = FunctionalFullyConnectedTensorProduct(
        "5x0e+1x0e+10x1e",
        "2x0e+2x1e+10x1e",
        "1000x1e+1000x0e",
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        gradient_normalization=gradient_normalization,
    )

    ws = [
        ins.weight_std * jax.random.normal(next(keys), ins.path_shape)
        for ins in tp.instructions
        if ins.has_weight
    ]
    x1 = e3nn.normal(tp.irreps_in1, next(keys), (), normalization=irrep_normalization)
    x2 = e3nn.normal(tp.irreps_in2, next(keys), (), normalization=irrep_normalization)

    v, s = tp.left_right(ws, x1, x2).chunks

    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(s**2)))) < 2.0
    if irrep_normalization == "component":
        assert jnp.exp(jnp.abs(jnp.log(jnp.mean(v**2)))) < 2.0
    if irrep_normalization == "norm":
        assert jnp.exp(jnp.abs(jnp.log(jnp.mean(jnp.sum(v**2, axis=1))))) < 2.0
