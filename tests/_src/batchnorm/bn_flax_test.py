import jax
import jax.numpy as jnp
import pytest
import numpy as np

import e3nn_jax as e3nn


@pytest.mark.parametrize(
    "irreps", [e3nn.Irreps("3x0e + 3x0o + 4x1e"), e3nn.Irreps("3x0o + 3x0e + 4x1e")]
)
def test_equivariant(keys, irreps):
    b = e3nn.flax.BatchNorm()

    variables = b.init(next(keys), e3nn.normal(irreps, next(keys), (16,)))
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    for _ in range(2):
        _, updates = b.apply(
            {"params": params, "batch_stats": batch_stats},
            e3nn.normal(irreps, next(keys), (16,)),
            use_running_average=False,
            mutable=["batch_stats"],
        )
        batch_stats = updates["batch_stats"]

    e3nn.utils.assert_equivariant(
        lambda x: b.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            use_running_average=False,
            mutable=["batch_stats"],
        )[0],
        next(keys),
        e3nn.normal(irreps, next(keys), (16,)),
    )
    e3nn.utils.assert_equivariant(
        lambda x: b.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            use_running_average=True,
            mutable=["batch_stats"],
        )[0],
        next(keys),
        e3nn.normal(irreps, next(keys), (16,)),
    )


def test_dtype(keys):
    jax.config.update("jax_enable_x64", True)

    b = e3nn.flax.BatchNorm()
    x = e3nn.normal("2x0e + 1e", next(keys), (128,))
    v = b.init(next(keys), x)

    e3nn.utils.assert_output_dtype_matches_input_dtype(
        lambda v, x: b.apply(v, x, use_running_average=False, mutable=["batch_stats"]),
        v,
        x,
    )
    e3nn.utils.assert_output_dtype_matches_input_dtype(
        lambda v, x: b.apply(v, x, use_running_average=True, mutable=["batch_stats"]),
        v,
        x,
    )


@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("affine", [True, False])
def test_modes_for_batchnorm(keys, affine, reduce, normalization):
    irreps = e3nn.Irreps("10x0e + 5x1e")

    b = e3nn.flax.BatchNorm(
        affine=affine, reduce=reduce, normalization=normalization, instance=False
    )

    variables = b.init(next(keys), e3nn.normal(irreps, next(keys), (20, 20)))
    params = variables["params"] if affine else dict()
    batch_stats = variables["batch_stats"]

    b.apply(
        {"params": params, "batch_stats": batch_stats},
        e3nn.normal(irreps, next(keys), (20, 20)),
        use_running_average=False,
        mutable=["batch_stats"],
    )
    b.apply(
        {"params": params, "batch_stats": batch_stats},
        e3nn.normal(irreps, next(keys), (20, 20)),
        use_running_average=True,
        mutable=["batch_stats"],
    )


@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("affine", [True, False])
def test_modes_for_instancenorm(keys, affine, reduce, normalization):
    irreps = e3nn.Irreps("10x0e + 5x1e")

    b = e3nn.flax.BatchNorm(
        affine=affine, reduce=reduce, normalization=normalization, instance=True
    )

    params = b.init(
        next(keys),
        e3nn.normal(irreps, next(keys), (20, 20)),
    )

    b.apply(params, e3nn.normal(irreps, next(keys), (20, 20)))


def test_mask(keys):
    irreps = e3nn.Irreps("0e")
    x = e3nn.normal(irreps, next(keys), (5,))
    m = jnp.array([True, True, True, False, False])

    b = e3nn.flax.BatchNorm(instance=False, momentum=1.0)
    variables = b.init(next(keys), x)

    y, updates = b.apply(
        variables, x, mask=m, use_running_average=False, mutable=["batch_stats"]
    )
    print(f"x = {x}")
    print(f"y = {y}")

    np.testing.assert_allclose(jnp.mean(y.array[:3]), 0.0, atol=1e-5, rtol=0.0)

    assert "batch_stats" in updates
    assert "mean" in updates["batch_stats"]
    assert "var" in updates["batch_stats"]

    np.testing.assert_allclose(
        updates["batch_stats"]["mean"], jnp.mean(x.array[:3]), atol=1e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        updates["batch_stats"]["var"], jnp.var(x.array[:3]), atol=1e-5, rtol=0.0
    )
