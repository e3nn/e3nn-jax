import jax


def jit_code(f, *args, **kwargs):
    """Jit a function with JAX. and return the jitted code as a string."""
    c = jax.xla_computation(f)(*args, **kwargs)
    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)

    import jaxlib.xla_extension as xla_ext

    option = xla_ext.HloPrintOptions.fingerprint()
    option.print_operand_shape = False
    option.print_result_shape = False
    option.print_program_shape = True

    code = e.hlo_modules()[0].to_string(option)
    code = code.split("ENTRY")[1]
    code = code.split("\n}")[0]
    code = "\n".join(x[2:] for x in code.split("\n")[1:])

    return code
