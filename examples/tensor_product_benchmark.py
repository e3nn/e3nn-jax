import argparse
import logging
import time

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, fully_connected_tensor_product

logging.basicConfig(level=logging.DEBUG)


# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        raise ValueError(str(arg))


def main():
    parser = argparse.ArgumentParser(
        prog="tensor_product_benchmark"
    )
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--irreps", type=str, default="8x0e + 8x1e + 8x2e + 8x3o")
    parser.add_argument("--irreps-in1", type=str, default=None)
    parser.add_argument("--irreps-in2", type=str, default=None)
    parser.add_argument("--irreps-out", type=str, default=None)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("--elementwise", action='store_true')
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    # device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    # args.cuda = device == 'cuda'

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("="*40)

    irreps_in1 = Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    irreps_in2 = Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    irreps_out = Irreps(args.irreps_out if args.irreps_out else args.irreps)

    if args.elementwise:
        # tp = ElementwiseTensorProduct(
        #     irreps_in1,
        #     irreps_in2,
        #     _specialized_code=args.specialized_code,
        #     _optimize_einsums=args.opt_ein
        # )
        # if args.backward:
        #     print("Elementwise TP has no weights, cannot backward. Setting --backward False.")
        #     args.backward = False
        pass
    else:
        instructions, nw, tp, _ = fully_connected_tensor_product(
            irreps_in1,
            irreps_in2,
            irreps_out,
            specialized_code=args.specialized_code,
            optimize_einsums=args.opt_ein
        )
        tp = jax.vmap(tp, (None, 0, 0), 0)
    # tp = tp.to(device=device)
    assert len(instructions) > 0, "Bad irreps, no instructions"
    # print(f"Tensor product: {tp}")
    print("Instructions:")
    for ins in instructions:
        print(f"  {ins}")

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    key = jax.random.PRNGKey(0)

    w = jax.random.normal(key, (nw,))
    inputs = iter([
        (
            irreps_in1.randn(key, (args.batch, -1)),
            irreps_in2.randn(key, (args.batch, -1))
        )
        for _ in range(args.n + warmup)
    ])

    if args.backward:
        # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
        tp_ = tp
        tp = jax.value_and_grad(lambda w, x1, x2: jnp.tanh(tp_(w, x1, x2)).sum(), 0)

    # compile
    if args.jit:
        tp = jax.jit(tp)

    print("starting...")

    for _ in range(warmup):
        z = tp(w, *next(inputs))
        jax.tree_map(lambda x: x.block_until_ready(), z)

    t = time.perf_counter()

    for _ in range(args.n):
        z = tp(w, *next(inputs))
        jax.tree_map(lambda x: x.block_until_ready(), z)

    perloop = (time.perf_counter() - t) / args.n

    print()
    print(f"{1e3 * perloop:.1f} ms")


if __name__ == '__main__':
    main()
