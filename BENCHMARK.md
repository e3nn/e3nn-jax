# TP Task

```
d/dw sum tanh TP(w, x1, x2)

x1, x2 = batch 64
TP = FullyConnected(irreps, irreps, irreps)
irreps = 128x0e + 128x1e + 128x2e
```

## Record on GTX1080
On Pytorch it takes 140 ms.

```
python examples/tensor_product_benchmark.py --irreps "128x0e + 128x1e + 128x2e" --extrachannels f --specialized-code f --fuse-all f --lists t --custom-einsum-vjp f --batch 64 -n 10

======= Benchmark with settings: ======
               jit : True
            irreps : 128x0e + 128x1e + 128x2e
        irreps_in1 : None
        irreps_in2 : None
        irreps_out : None
              cuda : True
          backward : True
           opt_ein : True
 custom_einsum_vjp : False
  specialized_code : False
       elementwise : False
     extrachannels : False
          fuse_all : False
             lists : True
                 n : 10
             batch : 64
========================================
31457280 parameters
starting...

12.9 ms
```

```
======= Benchmark with settings: ======
  specialized_code : True
             lists : False
========================================

12 ms
```

# QM9

## Record on V100
6 Conv + Gate
FC [64, 64]

lmax=2, mul=512
pytorch  430ms
jax      91ms
         x5

lmax=2, mul=128
pytorch  270ms
jax      42ms
         x6

lmax=1, mul=128
pytorch  100ms
jax      20ms
         x5

lmax=1, mul=256
pytorch  160ms
jax      34ms
         x5
