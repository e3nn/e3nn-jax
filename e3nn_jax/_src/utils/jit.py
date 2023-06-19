import jax


def jit_code(f, *args, **kwargs):
    """Jit a function with JAX. and return the jitted code as a string."""
    from jax.lib import xla_bridge
    import jaxlib.xla_extension as xla_ext

    f_jax = jax.jit(f)
    jax_comp = f_jax.lower(*args, **kwargs).compiler_ir(dialect="mhlo")
    jax_hlo = str(jax_comp)
    backend = xla_bridge.get_backend()
    jax_optimized_hlo = backend.compile(jax_hlo)

    option = xla_ext.HloPrintOptions.fingerprint()
    option.print_operand_shape = False
    option.print_result_shape = False
    option.print_program_shape = True
    code = jax_optimized_hlo.hlo_modules()[0].to_string(option)

    code = code.split("ENTRY")[1]
    code = code.split("\n}")[0]
    code = "\n".join(x[2:] for x in code.split("\n")[1:])

    return code
