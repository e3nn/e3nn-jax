from e3nn_jax import ScalarActivation, ElementwiseTensorProduct, Irreps, IrrepsData


class Gate:
    r"""Gate activation function.

    The gate activation is a direct sum of two sets of irreps. The first set
    of irreps is ``irreps_scalars`` passed through activation functions
    ``act_scalars``. The second set of irreps is ``irreps_gated`` multiplied
    by the scalars ``irreps_gates`` passed through activation functions
    ``act_gates``. Mathematically, this can be written as:

    .. math::
        \left(\bigoplus_i \phi_i(x_i) \right) \oplus \left(\bigoplus_j \phi_j(g_j) y_j \right)

    where :math:`x_i` and :math:`\phi_i` are from ``irreps_scalars`` and
    ``act_scalars``, and :math:`g_j`, :math:`\phi_j`, and :math:`y_j` are
    from ``irreps_gates``, ``act_gates``, and ``irreps_gated``.

    The parameters passed in should adhere to the following conditions:

    1. ``len(irreps_scalars) == len(act_scalars)``.
    2. ``len(irreps_gates) == len(act_gates)``.
    3. ``irreps_gates.num_irreps == irreps_gated.num_irreps``.

    Args:
        irreps_scalars (`Irreps`): The irreps of the scalars.
        act_scalars (list of functions): The activation functions of the scalars. The length of this list must be the same as the length of ``irreps_scalars``.
        irreps_gates (`Irreps`): The irreps of the gates.
        act_gates (list of functions): The activation functions of the gates. The length of this list must be the same as the length of ``irreps_gates``.
        irreps_gated (`Irreps`): The irreps multiplied by the gates.

    Returns:
        `Gate`: The gate activation function.

    Examples:
        >>> import jax.numpy as jnp
        >>> g = Gate("16x0o", [jnp.tanh], "32x0o", [jnp.tanh], "16x1e+16x1o")
        >>> g.irreps_out
        16x0o+16x1o+16x1e
    """
    irreps_in: Irreps
    irreps_out: Irreps

    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        super().__init__()
        irreps_scalars = Irreps(irreps_scalars)
        irreps_gates = Irreps(irreps_gates)
        irreps_gated = Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError((
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, "
                f"but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            ))

        # self.sc = _SortCut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = irreps_scalars, irreps_gates, irreps_gated  # self.sc.irreps_outs
        self.irreps_in = irreps_scalars + irreps_gates + irreps_gated

        self.act_scalars = ScalarActivation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = ScalarActivation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self.irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def __call__(self, features):
        r"""evaluate the gate activation function.

        Args:
            features: The features to be passed through the gate activation.

        Returns:
            `IrrepsData`: The output of the gate activation function.
        """
        features = IrrepsData.new(self.irreps_in, features).list
        scalars = IrrepsData.from_list(self.irreps_scalars, features[:len(self.irreps_scalars)])
        gates = IrrepsData.from_list(self.irreps_gates, features[len(self.irreps_scalars): -len(self.irreps_gated)])
        gated = IrrepsData.from_list(self.irreps_gated, features[-len(self.irreps_gated):])

        scalars = self.act_scalars(scalars)
        if gates:
            gates = self.act_gates(gates)
            gated = self.mul.left_right(gated, gates)
            features = IrrepsData.from_list(self.irreps_out, scalars.list + gated.list)
        else:
            features = scalars
        return features
