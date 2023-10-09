import jax
import pytest

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
