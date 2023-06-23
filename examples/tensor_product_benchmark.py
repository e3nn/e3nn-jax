import argparse
import re
import time

import haiku as hk
import jax
import jax.numpy as jnp
import jaxlib

import e3nn_jax as e3nn
from e3nn_jax._src.utils.jit import jit_code


# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(str(arg))


def main():
    parser = argparse.ArgumentParser(prog="tensor_product_benchmark")

    # Problem settings:
    parser.add_argument("--irreps", type=str, default="128x0e + 128x1e + 128x2e")
    parser.add_argument("--irreps-in1", type=str, default=None)
    parser.add_argument("--irreps-in2", type=str, default=None)
    parser.add_argument("--irreps-out", type=str, default=None)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--weights", type=t_or_f, default=True)
    parser.add_argument("--extrachannels", type=t_or_f, default=False)

    # Compilation settings:
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--lists", type=t_or_f, default=False)

    # Legacy Implementation settings:
    parser.add_argument("--module", type=t_or_f, default=False)
    parser.add_argument("--custom-einsum-jvp", type=t_or_f, default=False)
    parser.add_argument("--fused", type=t_or_f, default=False)
    parser.add_argument("--sparse", type=t_or_f, default=False)

    # Benchmark settings:
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    if not args.module:
        assert not args.custom_einsum_jvp
        assert not args.fused
        assert not args.sparse

    args.irreps_in1 = e3nn.Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    args.irreps_in2 = e3nn.Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    args.irreps_out = e3nn.Irreps(args.irreps_out if args.irreps_out else args.irreps)
    del args.irreps

    def k():
        k.key, x = jax.random.split(k.key)
        return x

    k.key = jax.random.PRNGKey(0)

    print("======= Versions: ======")
    print("jax:", jax.__version__)
    print("jaxlib:", jaxlib.__version__)
    print("e3nn_jax:", e3nn.__version__)

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("=" * 40)

    kwargs = dict(
        custom_einsum_jvp=args.custom_einsum_jvp,
        fused=args.fused,
        sparse=args.sparse,
    )

    @hk.without_apply_rng
    @hk.transform
    def tp(x1, x2):
        if args.module:
            assert not args.extrachannels
            assert args.weights
            return e3nn.haiku.FullyConnectedTensorProduct(args.irreps_out)(
                x1, x2, **kwargs
            )
        else:
            if args.extrachannels:
                assert not args.module
                x1 = x1.mul_to_axis()  # (batch, channels, irreps)
                x2 = x2.mul_to_axis()  # (batch, channels, irreps)
                x = e3nn.tensor_product(x1[..., :, None, :], x2[..., None, :, :])
                x = x.reshape(x.shape[:-3] + (-1,) + x.shape[-1:])
                x = x.axis_to_mul()
            else:
                x = e3nn.tensor_product(x1, x2)

            if args.weights:
                return e3nn.haiku.Linear(args.irreps_out)(x)
            else:
                return x.filter(keep=args.irreps_out)

    inputs = (
        e3nn.normal(args.irreps_in1, k(), (args.batch,)),
        e3nn.normal(args.irreps_in2, k(), (args.batch,)),
    )
    w = tp.init(k(), *inputs)

    # Ensure everything is on the GPU (shouldn't be necessary, but just in case)
    w, inputs = jax.tree_util.tree_map(jax.device_put, (w, inputs))

    print(f"{sum(x.size for x in jax.tree_util.tree_leaves(w))} parameters")

    f = tp.apply
    if args.lists:
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).chunks
    else:
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).array

    if args.backward:
        assert args.weights
        # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
        f_2 = f
        f = jax.value_and_grad(
            lambda w, x1, x2: sum(
                jnp.sum(jnp.tanh(out))
                for out in jax.tree_util.tree_leaves(f_2(w, x1, x2))
            )
        )

    # compile
    if args.jit:
        f = jax.jit(f)

    print("starting...")

    for _ in range(max(int(args.n // 100), 1)):
        z = f(w, *inputs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    print("output sum:", sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(z)))

    t = time.perf_counter()

    for _ in range(args.n):
        z = f(w, *inputs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    perloop = (time.perf_counter() - t) / args.n

    print()
    print(f"{1e3 * perloop:.2f} ms")

    with open("xla.txt", "wt") as file:
        code = jit_code(f, w, *inputs)
        code = re.sub(r"\d", "", code)

        file.write(code)


if __name__ == "__main__":
    main()
