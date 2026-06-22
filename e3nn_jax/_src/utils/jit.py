import jax


def jit_code(f, *args, **kwargs):
    """Jit a function with JAX and return the StableHLO MLIR code as a string."""
    f_jax = jax.jit(f)

    # Lower the function for the specific input shapes/types
    lowered = f_jax.lower(*args, **kwargs)

    # Extract the StableHLO MLIR representation
    # (You can also use dialect="mhlo" or dialect="hlo" if needed)
    mlir_module = lowered.compiler_ir(dialect="stablehlo")

    # Convert directly to string
    code = str(mlir_module)

    return code
