import argparse
import time

import haiku as hk
import jax
import jax.numpy as jnp
import jaxlib

import e3nn_jax as e3nn


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
    parser.add_argument("--irreps", type=str, default="128x0e + 128x1e + 128x2e")
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
    parser.add_argument("--module", type=t_or_f, default=False)
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    def k():
        k.key, x = jax.random.split(k.key)
        return x

    k.key = jax.random.PRNGKey(0)

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

    irreps_in1 = e3nn.Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    irreps_in2 = e3nn.Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    irreps_out = e3nn.Irreps(args.irreps_out if args.irreps_out else args.irreps)

    if args.elementwise:
        raise NotImplementedError
    elif args.extrachannels:

        @hk.without_apply_rng
        @hk.transform
        def tp(x1, x2):
            x = e3nn.tensor_product(
                x1[..., :, None, :],
                x2[..., None, :, :],
                custom_einsum_jvp=args.custom_einsum_jvp,
                fused=args.fused,
            )
            x = x.reshape(x.shape[:-3] + (-1,) + x.shape[-1:])
            x = x.axis_to_mul()
            return e3nn.Linear(irreps_out)(x)

        inputs = (e3nn.normal(irreps_in1, k(), (args.batch,)), e3nn.normal(irreps_in2, k(), (args.batch,)))
        inputs = (inputs[0].mul_to_axis(), inputs[1].mul_to_axis())

        f = tp.apply

    else:

        @hk.without_apply_rng
        @hk.transform
        def tp(x1, x2):
            if args.module:
                return e3nn.FullyConnectedTensorProduct(irreps_out)(
                    x1,
                    x2,
                    custom_einsum_jvp=args.custom_einsum_jvp,
                    fused=args.fused,
                )
            else:
                return e3nn.Linear(irreps_out)(
                    e3nn.tensor_product(
                        x1,
                        x2,
                        custom_einsum_jvp=args.custom_einsum_jvp,
                        fused=args.fused,
                    )
                )

        inputs = (e3nn.normal(irreps_in1, k(), (args.batch,)), e3nn.normal(irreps_in2, k(), (args.batch,)))

        f = tp.apply

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    w = tp.init(k(), *inputs)

    print(f"{sum(x.size for x in jax.tree_util.tree_leaves(w))} parameters")

    if args.lists:
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).list
    else:
        f_1 = f
        f = lambda w, x1, x2: f_1(w, x1, x2).array

    if args.backward:
        # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
        f_2 = f
        f = jax.value_and_grad(
            lambda w, x1, x2: sum(jnp.sum(jnp.tanh(out)) for out in jax.tree_util.tree_leaves(f_2(w, x1, x2)))
        )

    # compile
    if args.jit:
        f = jax.jit(f)

    print("starting...")

    for _ in range(warmup):
        z = f(w, *inputs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    t = time.perf_counter()

    for _ in range(args.n):
        z = f(w, *inputs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

    perloop = (time.perf_counter() - t) / args.n

    print()
    print(f"{1e3 * perloop:.1f} ms")

    c = jax.xla_computation(f)(w, *inputs)

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
