FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64 LIBRARY_PATH=/usr/local/cuda-11.6/lib64

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN git clone https://github.com/e3nn/e3nn-jax.git && cd e3nn-jax && git checkout 0.6.3 && python3 setup.py install && cd .. && rm -rf e3nn-jax

RUN pip3 install --upgrade nibabel wandb optax dm-haiku

# docker build -t mariogeiger/e3nn-jax:0.6.3-jax0.3.14-cuda11.6.2 .

# Test:
# docker run --gpus all -e CUDA_VISIBLE_DEVICES=7 --rm -v $(pwd):/home mariogeiger/e3nn-jax:0.6.3-jax0.3.14-cuda11.6.2 python3 /home/test.py
