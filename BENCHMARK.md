# TP Task

```
d/dw sum tanh TP(w, x1, x2)

x1, x2 = batch 64
TP = FullyConnected(irreps, irreps, irreps)
irreps = 128x0e + 128x1e + 128x2e
```

## Records on NVIDIA RTX A5500 using cuda 12.0

|version        |             | time                   |                             |
|---------------|-------------|------------------------|-----------------------------|
|jax 0.4.11     | e3nn 0.19.2 | 3.52ms                 | tensor_product -> Linear    | (improvement due to merge of two einsums, no more ui,vj->uvij followed by uvij,ijk->uvk)
|jax 0.4.11     | e3nn 0.19.2 | 3.57ms                 | FullyConnectedTensorProduct | (improvement due to merge of two einsums, no more ui,vj->uvij followed by uvij,ijk->uvk)
|jax 0.4.11     | e3nn 0.19.1 | 5.87ms                 | tensor_product -> Linear    |
|jax 0.4.11     | e3nn 0.19.1 | 4.58ms                 | FullyConnectedTensorProduct |

## Records on NVIDIA RTX A5000 using cuda 12.0

|version        |             | time                   |                             |
|---------------|-------------|------------------------|-----------------------------|
|jax 0.4.11     | e3nn 0.19.1 | 6.55ms                 | FullyConnectedTensorProduct |

## Records on NVIDIA RTX A5000 using cuda 11.7

|version        |             | time                   |                             |
|---------------|-------------|------------------------|-----------------------------|
|jax 0.3.25     | e3nn 0.12.0 | 5.2ms                  | FullyConnectedTensorProduct |
|jax 0.3.24     | e3nn 0.12.0 | 6.8ms                  | tensor_product -> Linear    |
|jax 0.3.24     | e3nn 0.12.0 | 5.2ms                  | FullyConnectedTensorProduct |
|jax 0.3.24     | e3nn 0.7.0  | 5.2ms                  | FullyConnectedTensorProduct |
|jax 0.3.24     | e3nn 0.6.0  | 5.2ms                  | FullyConnectedTensorProduct |
|jax 0.3.24     | e3nn 0.4.0  | 5.2ms                  | FullyConnectedTensorProduct |
|jax 0.3.15     | e3nn 0.12.0 | 5.2ms                  | FullyConnectedTensorProduct |

## Records on NVIDIA RTX A5000 using cuda 11.6

|version        |            | time                   |
|---------------|------------|------------------------|
|pytorch 1.11.0 | e3nn 0.5.0 | between 13ms and 14ms. |
|jax 0.3.15     | e3nn 0.7.0 | 1.7ms                  | (probably not correct)

x8 speedup

## Records on GTX1080
On Pytorch it takes 140 ms.

```
python examples/tensor_product_benchmark.py --irreps "128x0e + 128x1e + 128x2e" --extrachannels f --specialized-code f --fused f --lists t --custom-einsum-jvp f --batch 64 -n 10

======= Benchmark with settings: ======
               jit : True
            irreps : 128x0e + 128x1e + 128x2e
        irreps_in1 : None
        irreps_in2 : None
        irreps_out : None
              cuda : True
          backward : True
           opt_ein : True
 custom_einsum_jvp : False
  specialized_code : False
       elementwise : False
     extrachannels : False
          fused : False
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
pytorch  230ms
jax      42ms
         x5

lmax=1, mul=128
pytorch  100ms
jax      20ms
         x5

lmax=1, mul=256
pytorch  130ms
jax      34ms
         x4
