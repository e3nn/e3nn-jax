import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import FunctionalFullyConnectedTensorProduct, FunctionalTensorProduct, FunctionalTensorSquare, Irreps
from e3nn_jax.util import prod


@pytest.mark.parametrize("connection_mode", ["uvw", "uvu", "uvv"])
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("optimize_einsums", [False, True])
@pytest.mark.parametrize("specialized_code", [False, True])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm"])
def test_modes(keys, irrep_normalization, specialized_code, optimize_einsums, jitted, connection_mode):
    tp = FunctionalTensorProduct(
        Irreps("10x0o + 10x1o + 1x2e"),
        Irreps("10x0o + 10x1o + 1x2o"),
        Irreps("10x0e + 10x1e + 2x2o"),
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
            specialized_code=specialized_code,
            optimize_einsums=optimize_einsums,
            custom_einsum_vjp=optimize_einsums,
        )

    if jitted:
        f = jax.jit(f)

    g = tp.left_right

    ws = [jax.random.normal(next(keys), ins.path_shape) for ins in tp.instructions if ins.has_weight]
    x1 = tp.irreps_in1.randn(next(keys), (-1,), normalization=irrep_normalization)
    x2 = tp.irreps_in2.randn(next(keys), (-1,), normalization=irrep_normalization)

    a = f(ws, x1, x2).array
    b = g(ws, x1, x2).array
    assert jnp.allclose(a, b, rtol=1e-4, atol=1e-6), jnp.max(jnp.abs(a - b))


def test_fuse_all(keys):
    tp = FunctionalTensorProduct(
        "10x0e + 5x1e",
        "0e + 1e",
        "10x0e + 5x1e",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 1, "uvu", True),
            (1, 0, 1, "uvu", True),
        ],
    )
    w = [jax.random.normal(keys[1], ins.path_shape) for ins in tp.instructions]
    x = jax.random.normal(keys[2], (25,))
    y = jax.random.normal(keys[3], (4,))

    assert jnp.allclose(
        tp.left_right(w, x, y, fuse_all=True).array,
        tp.left_right(w, x, y, fuse_all=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fuse_all_no_weight(keys):
    tp = FunctionalTensorProduct(
        "10x0e",
        "10x0e",
        "10x0e",
        [
            (0, 0, 0, "uuu", False),
        ],
    )
    w = jnp.ones(0)
    x = jax.random.normal(keys[2], (10,))
    y = jax.random.normal(keys[3], (10,))

    assert jnp.allclose(
        tp.left_right(w, x, y, fuse_all=True).array,
        tp.left_right(w, x, y, fuse_all=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fuse_all_mix_weight(keys):
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
    x = jax.random.normal(keys[2], (5,))
    y = jax.random.normal(keys[3], (5,))

    assert jnp.allclose(
        tp.left_right(w, x, y, fuse_all=True).array,
        tp.left_right(w, x, y, fuse_all=False).array,
        rtol=1e-4,
        atol=1e-6,
    )


def test_fuse(keys):
    tp = FunctionalFullyConnectedTensorProduct("2x0e+1e", "0e+1e", "1e+0e")

    ws = [jax.random.normal(next(keys), ins.path_shape) for ins in tp.instructions if ins.has_weight]
    wf = jnp.concatenate([w.flatten() for w in ws])
    x1 = tp.irreps_in1.randn(next(keys), (-1,))
    x2 = tp.irreps_in2.randn(next(keys), (-1,))

    a = tp.left_right(ws, x1, x2, fuse_all=False).array
    b = tp.left_right(wf, x1, x2, fuse_all=True).array
    assert jnp.allclose(a, b, rtol=1e-4, atol=1e-6), (a, b)


@pytest.mark.parametrize("gradient_normalization", ["element", "path", 0.5])
@pytest.mark.parametrize("path_normalization", ["element", "path", 0.5])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm"])
def test_normalization(keys, irrep_normalization, path_normalization, gradient_normalization):
    tp = FunctionalFullyConnectedTensorProduct(
        "5x0e+1x0e+10x1e",
        "2x0e+2x1e+10x1e",
        "1000x1e+1000x0e",
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        gradient_normalization=gradient_normalization,
    )

    ws = [ins.weight_std * jax.random.normal(next(keys), ins.path_shape) for ins in tp.instructions if ins.has_weight]
    x1 = tp.irreps_in1.randn(next(keys), (-1,), normalization=irrep_normalization)
    x2 = tp.irreps_in2.randn(next(keys), (-1,), normalization=irrep_normalization)

    v, s = tp.left_right(ws, x1, x2).list

    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(s**2)))) < 2.0
    if irrep_normalization == "component":
        assert jnp.exp(jnp.abs(jnp.log(jnp.mean(v**2)))) < 2.0
    if irrep_normalization == "norm":
        assert jnp.exp(jnp.abs(jnp.log(jnp.mean(jnp.sum(v**2, axis=1))))) < 2.0


def test_square_normalization(keys):
    irreps = Irreps("2x0e + 3x1e + 2x2e + 3e")
    tp = FunctionalTensorSquare(irreps, irreps, irrep_normalization="component")
    n = sum(prod(ins.path_shape) for ins in tp.instructions if ins.has_weight)

    @jax.vmap
    def f(w, x):
        return tp.left_right(w, x, x).array

    k = 1_000_000
    w = jax.random.normal(keys[0], (k, n))
    x = irreps.randn(keys[1], (k, -1), normalization="component")
    y = f(w, x)
    assert jnp.all(jnp.exp(jnp.abs(jnp.log(jnp.mean(y**2, 0)))) < 1.1)
