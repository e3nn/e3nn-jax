__conf = {
    "irrep_normalization": "component",  # "component" or "norm"
    "path_normalization": "element",  # "element" or "path"
    "gradient_normalization": "path",  # "element", "path" or float,
    # "element" is the default in modules provided by pytorch/haiku
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
