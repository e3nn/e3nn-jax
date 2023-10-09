try:
    import flax

    del flax
except ImportError:
    pass
else:
    from e3nn_jax._src.linear_flax import Linear
    from e3nn_jax._src.mlp_flax import MultiLayerPerceptron
    from e3nn_jax._src.batchnorm.bn_flax import BatchNorm

    __all__ = ["Linear", "MultiLayerPerceptron", "BatchNorm"]
