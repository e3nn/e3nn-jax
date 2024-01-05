try:
    import equinox

    del equinox
except ImportError:
    pass
else:
    from e3nn_jax._src.linear_equinox import Linear

    __all__ = ["Linear"]
