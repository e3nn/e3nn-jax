__conf = {
    "irrep_normalization": "component",
    "path_normalization": "element",
    "gradient_normalization": "path",
    "spherical_harmonics_algorithm": "automatic",
    "spherical_harmonics_normalization": "component",
    "specialized_code": False,
    "optimize_einsums": True,
    "custom_einsum_vjp": False,
    "fuse_all": False,
}


def config(name, value=None):
    if value is None:
        return __conf[name]

    if name in __conf:
        __conf[name] = value
    else:
        raise ValueError("Unknown configuration option: {}".format(name))
