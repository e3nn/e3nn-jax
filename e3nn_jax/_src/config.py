__conf = {
    "irrep_normalization": "component",
    "path_normalization": "element",
    "gradient_normalization": "path",
    "spherical_harmonics_algorithm": "automatic",
    "spherical_harmonics_normalization": "component",
    "custom_einsum_jvp": False,
    "fused": False,
}


def config(name, value=None):
    if value is None:
        return __conf[name]

    if name in __conf:
        __conf[name] = value
    else:
        raise ValueError("Unknown configuration option: {}".format(name))
