import argparse

# import logging
import math
import time
from functools import partial, reduce

import jax
import jax.numpy as jnp
import jaxlib
import e3nn_jax as e3nn
from e3nn_jax import FunctionalFullyConnectedTensorProduct, Irreps, IrrepsArray

# logging.basicConfig(level=logging.DEBUG)


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
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--irreps", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--irreps-in1", type=str, default=None)
    parser.add_argument("--irreps-in2", type=str, default=None)
    parser.add_argument("--irreps-out", type=str, default=None)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--custom-einsum-jvp", type=t_or_f, default=False)
    parser.add_argument("--elementwise", type=t_or_f, default=False)
    parser.add_argument("--extrachannels", type=t_or_f, default=False)
    parser.add_argument("--fused", type=t_or_f, default=False)
    parser.add_argument("--lists", type=t_or_f, default=False)
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    # device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    # args.cuda = device == 'cuda'
    print("======= Versions: ======")
    print("jax:", jax.__version__)
    print("jaxlib:", jaxlib.__version__)
    print("e3nn_jax:", e3nn.__version__)

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("=" * 40)

    irreps_in1 = Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    irreps_in2 = Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    irreps_out = Irreps(args.irreps_out if args.irreps_out else args.irreps)

    if args.elementwise:
        pass
    elif args.extrachannels:

        def compose(f, g):
            return lambda *x: g(f(*x))

        c_in1 = reduce(math.gcd, [mul for mul, ir in irreps_in1])
        c_in2 = reduce(math.gcd, [mul for mul, ir in irreps_in2])
        c_out = reduce(math.gcd, [mul for mul, ir in irreps_out])

        irreps_in1_red = Irreps([(mul // c_in1, ir) for mul, ir in irreps_in1])
        irreps_in2_red = Irreps([(mul // c_in2, ir) for mul, ir in irreps_in2])
        irreps_out_red = Irreps([(mul // c_out, ir) for mul, ir in irreps_out])

        tp = FunctionalFullyConnectedTensorProduct(irreps_in1_red, irreps_in2_red, irreps_out_red)

        f = partial(
            tp.left_right,
            custom_einsum_jvp=args.custom_einsum_jvp,
            fused=args.fused,
        )

        f = jax.vmap(f, (0, None, None), 0)  # channel_out
        f = jax.vmap(f, (0, None, 0), 0)  # channel_in2
        f = jax.vmap(f, (0, 0, None), 0)  # channel_in1
        f = compose(f, lambda z: jnp.sum(z, (0, 1)) / jnp.sqrt(z.shape[0] * z.shape[1]))

        f__ = f

        def f(w, x1, x2):
            return f__(w, x1.reshape(c_in1, irreps_in1_red.dim), x2.reshape(c_in2, irreps_in2_red.dim))

        tp.left_right = f

        w_shape = (c_in1, c_in2, c_out)
        print(f"extrachannels = {w_shape}")
    else:
        tp = FunctionalFullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
        )

        w_shape = ()

        f = partial(
            tp.left_right,
            custom_einsum_jvp=args.custom_einsum_jvp,
            fused=args.fused,
        )
    f = jax.vmap(f, (None, 0, 0), 0)

    assert len(tp.instructions) > 0, "Bad irreps, no instructions"

    print("Instructions:")
    for ins in tp.instructions:
        print(f"  {ins}")

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    def k():
        k.key, x = jax.random.split(k.key)
        return x

    k.key = jax.random.PRNGKey(0)

    ws = [jax.random.normal(k(), w_shape + ins.path_shape) for ins in tp.instructions]

    if args.fused:
        ws = jnp.concatenate([w.reshape(w_shape + (-1,)) for w in ws], axis=-1)
        print(f"flat weight shape = {ws.shape}")

    print(f"{sum(x.size for x in jax.tree_util.tree_leaves(ws))} parameters")

    if args.lists:
        inputs = iter(
            [
                (
                    IrrepsArray(irreps_in1, irreps_in1.randn(k(), (args.batch, -1))).list,
                    IrrepsArray(irreps_in2, irreps_in2.randn(k(), (args.batch, -1))).list,
                )
                for _ in range(args.n + warmup)
            ]
        )
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).list
    else:
        inputs = iter(
            [
                (irreps_in1.randn(k(), (args.batch, -1)), irreps_in2.randn(k(), (args.batch, -1)))
                for _ in range(args.n + warmup)
            ]
        )
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).array

    if args.backward:
        # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
        f_2 = f
        f = jax.value_and_grad(
            lambda ws, x1, x2: sum(jnp.sum(jnp.tanh(x)) for x in jax.tree_util.tree_leaves(f_2(ws, x1, x2))), 0
        )

    # compile
    if args.jit:
        f = jax.jit(f)

    print("starting...")

    for _ in range(warmup):
        z = f(ws, *next(inputs))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    t = time.perf_counter()

    for _ in range(args.n):
        z = f(ws, *next(inputs))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    perloop = (time.perf_counter() - t) / args.n

    print()
    print(f"{1e3 * perloop:.1f} ms")

    x1 = irreps_in1.randn(k(), (args.batch, -1))
    x2 = irreps_in2.randn(k(), (args.batch, -1))

    c = jax.xla_computation(f)(ws, x1, x2)

    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)
    import jaxlib.xla_extension as xla_ext

    option = xla_ext.HloPrintOptions.fingerprint()
    option.print_operand_shape = False
    option.print_result_shape = False
    option.print_program_shape = True

    with open("xla.txt", "wt") as f:
        f.write(e.hlo_modules()[0].to_string(option))


if __name__ == "__main__":
    main()
