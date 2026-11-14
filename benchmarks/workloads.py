"""Registry of benchmark workloads.

Irreps and call patterns mirror real models built on e3nn-jax: NequIP- and
MACE-style message passing as implemented in ``tensorial`` (per-edge tensor
product with spherical harmonics, per-irrep radial weights, scatter-add,
flax ``Linear`` and gates).

Each workload factory takes ``(size, dtype, key)`` and returns
``(fn, diff_args, nondiff_args)`` where ``fn(*diff_args, *nondiff_args)``
returns an ``IrrepsArray`` (or array). ``diff_args`` are differentiated in
"bwd" mode; ``nondiff_args`` (e.g. integer indices) are not.
"""

import dataclasses
from typing import Callable

import flax.linen as nn
import haiku as hk
import jax

import e3nn_jax as e3nn

# Model-like irreps configurations.
CONFIGS = {
    # NequIP: 32 channels, lmax 2 features, lmax 2 spherical harmonics.
    "nequip": dict(
        features="32x0e + 32x1o + 32x2e",
        sh_lmax=2,
        out="32x0e + 32x1o + 32x2e",
    ),
    # NequIP-large: 64 channels, both parities, lmax 3.
    "nequip_l3": dict(
        features="64x0e + 64x0o + 64x1e + 64x1o + 64x2e + 64x2o + 64x3e + 64x3o",
        sh_lmax=3,
        out="64x0e + 64x0o + 64x1e + 64x1o + 64x2e + 64x2o + 64x3e + 64x3o",
    ),
    # MACE: 128 channels, scalars + vectors, lmax 3 spherical harmonics.
    "mace": dict(
        features="128x0e + 128x1o",
        sh_lmax=3,
        out="128x0e + 128x1o + 128x2e + 128x3o",
    ),
}


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    make: Callable
    sizes: tuple[int, ...]
    description: str


REGISTRY: dict[str, Workload] = {}


def register(name: str, sizes: tuple[int, ...], description: str):
    def deco(make):
        assert name not in REGISTRY, name
        REGISTRY[name] = Workload(name, make, sizes, description)
        return make

    return deco


def _keys(key, n):
    return list(jax.random.split(key, n))


def _vectors(key, n, dtype):
    return jax.random.normal(key, (n, 3), dtype)


def _sh(cfg, key, n, dtype):
    vec = e3nn.IrrepsArray("1o", _vectors(key, n, dtype))
    return e3nn.spherical_harmonics(range(cfg["sh_lmax"] + 1), vec, normalize=True)


def _filter_out(cfg):
    return e3nn.Irreps(cfg["out"]) + "0e"


# --------------------------------------------------------------------------
# Micro workloads
# --------------------------------------------------------------------------

for _cfg_name, _cfg in CONFIGS.items():

    def _make_mul_weights(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        x = e3nn.normal(cfg["features"], k1, (n,), dtype=dtype)
        w = jax.random.normal(k2, (n, x.irreps.num_irreps), dtype)
        return (lambda x, w: x * w), (x, w), ()

    register(
        f"mul_scalar_weights/{_cfg_name}",
        (100_000,),
        "IrrepsArray * per-irrep weights (radial weights on messages)",
    )(_make_mul_weights)

    def _make_tp_sh(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        x = e3nn.normal(cfg["features"], k1, (n,), dtype=dtype)
        y = _sh(cfg, k2, n, dtype)
        filt = _filter_out(cfg)
        return (lambda x, y: e3nn.tensor_product(x, y, filter_ir_out=filt)), (x, y), ()

    register(
        f"tp_features_sh/{_cfg_name}",
        (10_000, 100_000),
        "tensor_product(edge features, sh, filter_ir_out=out+0e)",
    )(_make_tp_sh)

    def _make_conv_messages(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        x = e3nn.normal(cfg["features"], k1, (n,), dtype=dtype)
        y = _sh(cfg, k2, n, dtype)
        filt = _filter_out(cfg)

        def f(x, y):
            tp = e3nn.tensor_product(x, y, filter_ir_out=filt)
            return e3nn.concatenate([x.filter(filt), tp]).regroup()

        return f, (x, y), ()

    register(
        f"tp_concat_regroup/{_cfg_name}",
        (100_000,),
        "tensor_product + concatenate([x.filter, tp]).regroup() (message construction)",
    )(_make_conv_messages)

    def _make_linear(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        x = e3nn.normal(cfg["out"], k1, (n,), dtype=dtype)
        lin = e3nn.flax.Linear(cfg["out"])
        params = lin.init(k2, x)
        params = jax.tree.map(lambda p: p.astype(dtype), params)
        return (lambda p, x: lin.apply(p, x)), (params, x), ()

    register(
        f"linear_flax/{_cfg_name}",
        (10_000, 100_000),
        "e3nn.flax.Linear(out)(x) on node features",
    )(_make_linear)

    def _make_scatter(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        n_nodes = max(n // 20, 1)
        x = e3nn.normal(cfg["out"], k1, (n,), dtype=dtype)
        receivers = jax.random.randint(k2, (n,), 0, n_nodes)

        def f(x, receivers):
            return e3nn.zeros(x.irreps, (n_nodes,), x.dtype).at[receivers].add(x)

        return f, (x,), (receivers,)

    register(
        f"scatter_add/{_cfg_name}",
        (100_000,),
        "zeros(nodes).at[receivers].add(edge messages), ~20 edges per node",
    )(_make_scatter)

    def _make_gather(n, dtype, key, cfg=_cfg):
        k1, k2 = _keys(key, 2)
        n_nodes = max(n // 20, 1)
        x = e3nn.normal(cfg["features"], k1, (n_nodes,), dtype=dtype)
        senders = jax.random.randint(k2, (n,), 0, n_nodes)
        return (lambda x, s: x[s]), (x,), (senders,)

    register(
        f"gather/{_cfg_name}",
        (100_000,),
        "node_feats[senders]",
    )(_make_gather)


for _lmax in (2, 3, 6):

    def _make_sh(n, dtype, key, lmax=_lmax):
        vec = e3nn.IrrepsArray("1o", _vectors(key, n, dtype))
        return (
            (lambda v: e3nn.spherical_harmonics(range(lmax + 1), v, normalize=True)),
            (vec,),
            (),
        )

    register(
        f"spherical_harmonics/l{_lmax}",
        (100_000,),
        "spherical_harmonics(range(lmax+1), vectors, normalize=True)",
    )(_make_sh)


for _mul in (8, 32):

    def _make_square(n, dtype, key, mul=_mul):
        x = e3nn.normal(f"{mul}x0e + {mul}x1o + {mul}x2e", key, (n,), dtype=dtype)
        return e3nn.tensor_square, (x,), ()

    register(
        f"tensor_square/mul{_mul}",
        (1_000, 10_000),
        "tensor_square(mul x (0e+1o+2e))",
    )(_make_square)


def _make_tp_axis(n, dtype, key):
    # MACE-style product basis term: channels as a leading axis, mul 1 irreps.
    k1, k2 = _keys(key, 2)
    irreps = "0e + 1o + 2e + 3o"
    x = e3nn.normal(irreps, k1, (n, 32), dtype=dtype)
    y = e3nn.normal(irreps, k2, (n, 32), dtype=dtype)
    filt = e3nn.Irreps("0e + 1o + 2e + 3o")
    return (lambda x, y: e3nn.tensor_product(x, y, filter_ir_out=filt)), (x, y), ()


register(
    "tp_channels_axis/mace",
    (1_000, 10_000),
    "tensor_product on (n, channels, 0e+1o+2e+3o) arrays (MACE product basis)",
)(_make_tp_axis)


def _make_fctp_haiku(n, dtype, key):
    # Yardstick: fused weighted TP (legacy). Not an optimisation target.
    cfg = CONFIGS["nequip"]
    k1, k2, k3 = _keys(key, 3)
    x = e3nn.normal(cfg["features"], k1, (n,), dtype=dtype)
    y = _sh(cfg, k2, n, dtype)

    @hk.without_apply_rng
    @hk.transform
    def mod(x, y):
        return e3nn.haiku.FullyConnectedTensorProduct(cfg["out"])(x, y)

    params = mod.init(k3, x, y)
    params = jax.tree.map(lambda p: p.astype(dtype), params)
    return (lambda p, x, y: mod.apply(p, x, y)), (params, x, y), ()


register(
    "yardstick_fctp_haiku/nequip",
    (10_000,),
    "haiku.FullyConnectedTensorProduct (legacy, fused weights) - reference only",
)(_make_fctp_haiku)


def _make_tp_linear(n, dtype, key):
    # Recommended replacement for the yardstick: tensor_product -> Linear.
    cfg = CONFIGS["nequip"]
    k1, k2, k3 = _keys(key, 3)
    x = e3nn.normal(cfg["features"], k1, (n,), dtype=dtype)
    y = _sh(cfg, k2, n, dtype)
    lin = e3nn.flax.Linear(cfg["out"])
    params = lin.init(k3, e3nn.tensor_product(x, y))
    params = jax.tree.map(lambda p: p.astype(dtype), params)
    return (lambda p, x, y: lin.apply(p, e3nn.tensor_product(x, y))), (params, x, y), ()


register(
    "tp_then_linear/nequip",
    (10_000,),
    "flax.Linear(out)(tensor_product(x, sh)) - recommended form of the yardstick",
)(_make_tp_linear)


# --------------------------------------------------------------------------
# Macro workload: NequIP-style interaction block (as in tensorial)
# --------------------------------------------------------------------------


class _Convolution(nn.Module):
    irreps_out: str
    radial_neurons: int = 64

    @nn.compact
    def __call__(self, node_feats, sh, radial, senders, receivers):
        irreps_out = e3nn.Irreps(self.irreps_out)
        filt = irreps_out + "0e"
        messages = node_feats[senders]
        tp = e3nn.tensor_product(messages, sh, filter_ir_out=filt)
        messages = e3nn.concatenate([messages.filter(filt), tp]).regroup()
        weights = e3nn.flax.MultiLayerPerceptron(
            (self.radial_neurons, self.radial_neurons, messages.irreps.num_irreps),
            jax.nn.silu,
            with_bias=False,
            output_activation=False,
        )(radial)
        messages = messages * weights
        zeros = e3nn.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)
        return node_feats / 20.0**0.5


class _InteractionBlock(nn.Module):
    irreps_out: str

    @nn.compact
    def __call__(self, node_feats, sh, radial, senders, receivers):
        irreps_out = e3nn.Irreps(self.irreps_out)
        num_gates = sum(mul for mul, ir in irreps_out if not ir.is_scalar())
        x = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)
        x = _Convolution(self.irreps_out)(x, sh, radial, senders, receivers)
        x = e3nn.flax.Linear(irreps_out + f"{num_gates}x0e", name="linear_down")(x)
        return e3nn.gate(x)


for _cfg_name in ("nequip", "mace"):

    def _make_block(n, dtype, key, cfg=CONFIGS[_cfg_name]):
        k = _keys(key, 6)
        n_nodes = max(n // 20, 1)
        node_feats = e3nn.normal(cfg["features"], k[0], (n_nodes,), dtype=dtype)
        sh = _sh(cfg, k[1], n, dtype)
        radial = jax.random.normal(k[2], (n, 8), dtype)
        senders = jax.random.randint(k[3], (n,), 0, n_nodes)
        receivers = jax.random.randint(k[4], (n,), 0, n_nodes)
        block = _InteractionBlock(cfg["out"])
        params = block.init(k[5], node_feats, sh, radial, senders, receivers)
        params = jax.tree.map(lambda p: p.astype(dtype), params)

        def f(params, node_feats, sh, radial, senders, receivers):
            return block.apply(params, node_feats, sh, radial, senders, receivers)

        return f, (params, node_feats, sh, radial), (senders, receivers)

    register(
        f"macro_interaction_block/{_cfg_name}",
        (20_000, 100_000),
        "Linear -> [gather, TP(sh), concat+regroup, radial MLP weights, scatter] -> Linear -> gate",
    )(_make_block)


# --------------------------------------------------------------------------
# Reference task from docs/BENCHMARK.rst ("TP Task")
#   d/dw sum tanh TP(w, x1, x2); batch 64; irreps 128x0e + 128x1e + 128x2e
# Same computation as examples/tensor_product_benchmark.py. In "bwd" mode only
# the weights are differentiated (x1, x2 are passed as non-diff args), exactly
# like the docs/example.
# --------------------------------------------------------------------------

_DOCS_IRREPS = "128x0e + 128x1e + 128x2e"


def _make_docs_tp_linear(n, dtype, key):
    k1, k2, k3 = _keys(key, 3)
    x1 = e3nn.normal(_DOCS_IRREPS, k1, (n,), dtype=dtype)
    x2 = e3nn.normal(_DOCS_IRREPS, k2, (n,), dtype=dtype)

    @hk.without_apply_rng
    @hk.transform
    def mod(x1, x2):
        return e3nn.haiku.Linear(_DOCS_IRREPS)(e3nn.tensor_product(x1, x2))

    params = jax.tree.map(lambda p: p.astype(dtype), mod.init(k3, x1, x2))
    return mod.apply, (params,), (x1, x2)


register(
    "docs_tp_task/tp_then_linear",
    (64,),
    "docs/BENCHMARK.rst TP task: haiku.Linear(tensor_product(x1, x2)), d/dw",
)(_make_docs_tp_linear)


def _make_docs_fctp(n, dtype, key):
    k1, k2, k3 = _keys(key, 3)
    x1 = e3nn.normal(_DOCS_IRREPS, k1, (n,), dtype=dtype)
    x2 = e3nn.normal(_DOCS_IRREPS, k2, (n,), dtype=dtype)

    @hk.without_apply_rng
    @hk.transform
    def mod(x1, x2):
        return e3nn.haiku.FullyConnectedTensorProduct(_DOCS_IRREPS)(x1, x2)

    params = jax.tree.map(lambda p: p.astype(dtype), mod.init(k3, x1, x2))
    return mod.apply, (params,), (x1, x2)


register(
    "docs_tp_task/fully_connected_tp",
    (64,),
    "docs/BENCHMARK.rst TP task: haiku.FullyConnectedTensorProduct(x1, x2), d/dw",
)(_make_docs_fctp)
