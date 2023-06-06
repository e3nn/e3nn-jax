import warnings
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.irreps import IntoIrreps
from e3nn_jax._src.irreps_array import _infer_backend, _standardize_axis


def _align_two_irreps_arrays(
    input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray
) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
    assert input1.irreps.num_irreps == input2.irreps.num_irreps

    irreps_in1 = list(input1.irreps)
    irreps_in2 = list(input2.irreps)

    i = 0
    while i < min(len(irreps_in1), len(irreps_in2)):
        mul_1, ir_1 = irreps_in1[i]
        mul_2, ir_2 = irreps_in2[i]

        if mul_1 < mul_2:
            irreps_in2[i] = (mul_1, ir_2)
            irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

        if mul_2 < mul_1:
            irreps_in1[i] = (mul_2, ir_1)
            irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))

        i += 1

    input1 = input1._convert(irreps_in1)
    input2 = input2._convert(irreps_in2)

    assert [mul for mul, _ in input1.irreps] == [mul for mul, _ in input2.irreps]
    return input1, input2


def _reduce(
    op,
    array: e3nn.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> e3nn.IrrepsArray:
    axis = _standardize_axis(axis, array.ndim)

    if axis == ():
        return array

    if axis[-1] < array.ndim - 1:
        # irrep dimension is not affected by mean
        return e3nn.IrrepsArray(
            array.irreps,
            op(array.array, axis=axis, keepdims=keepdims),
            [
                None if x is None else op(x, axis=axis, keepdims=keepdims)
                for x in array.list
            ],
        )

    array = _reduce(op, array, axis=axis[:-1], keepdims=keepdims)
    return e3nn.IrrepsArray.from_list(
        e3nn.Irreps([(1, ir) for _, ir in array.irreps]),
        [None if x is None else op(x, axis=-2, keepdims=True) for x in array.list],
        array.shape[:-1],
        array.dtype,
    )


def mean(
    array: e3nn.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> e3nn.IrrepsArray:
    """Mean of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the mean is computed.

    Returns:
        `IrrepsArray`: mean of the input array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0e", jnp.arange(2 * 5).reshape(2, 5))
        >>> e3nn.mean(x, axis=0)
        3x0e+2x0e [2.5 3.5 4.5 5.5 6.5]
        >>> e3nn.mean(x, axis=1)
        1x0e+1x0e
        [[1.  3.5]
         [6.  8.5]]
        >>> e3nn.mean(x)
        1x0e+1x0e [3.5 6. ]
    """
    jnp = _infer_backend(array.array)
    return _reduce(jnp.mean, array, axis, keepdims)


def sum_(
    array: e3nn.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> e3nn.IrrepsArray:
    """Sum of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the sum is computed.

    Returns:
        `IrrepsArray`: sum of the input array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0e", jnp.arange(2 * 5).reshape(2, 5))
        >>> e3nn.sum(x, axis=0)
        3x0e+2x0e [ 5  7  9 11 13]
        >>> e3nn.sum(x, axis=1)
        1x0e+1x0e
        [[ 3  7]
         [18 17]]
        >>> e3nn.sum(x)
        1x0e+1x0e [21 24]
        >>> e3nn.sum(x.regroup())
        1x0e [45]
    """
    jnp = _infer_backend(array.array)
    return _reduce(jnp.sum, array, axis, keepdims)


def concatenate(arrays: List[e3nn.IrrepsArray], axis: int = -1) -> e3nn.IrrepsArray:
    r"""Concatenate a list of IrrepsArray.

    Args:
        arrays (list of `IrrepsArray`): list of data to concatenate
        axis (int): axis to concatenate on

    Returns:
        `IrrepsArray`: concatenated array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0o", jnp.arange(2 * 5).reshape(2, 5))
        >>> y = e3nn.IrrepsArray("3x0e + 2x0o", jnp.arange(2 * 5).reshape(2, 5) + 10)
        >>> e3nn.concatenate([x, y], axis=0)
        3x0e+2x0o
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> e3nn.concatenate([x, y], axis=1)
        3x0e+2x0o+3x0e+2x0o
        [[ 0  1  2  3  4 10 11 12 13 14]
         [ 5  6  7  8  9 15 16 17 18 19]]
    """
    if len(arrays) == 0:
        raise ValueError("Cannot concatenate empty list of IrrepsArray")

    arrays = [e3nn.IrrepsArray.as_irreps_array(x) for x in arrays]
    axis = _standardize_axis(axis, arrays[0].ndim)[0]

    jnp = _infer_backend([x.array for x in arrays])

    if axis == arrays[0].ndim - 1:
        irreps = e3nn.Irreps(sum([x.irreps for x in arrays], e3nn.Irreps("")))
        return e3nn.IrrepsArray(
            irreps=irreps,
            array=jnp.concatenate([x.array for x in arrays], axis=-1),
            list=sum([x.list for x in arrays], []),
        )

    if {x.irreps for x in arrays} != {arrays[0].irreps}:
        raise ValueError("Irreps must be the same for all arrays")

    arrays = [
        x.replace_none_with_zeros() for x in arrays
    ]  # TODO this could be optimized
    return e3nn.IrrepsArray(
        irreps=arrays[0].irreps,
        array=jnp.concatenate([x.array for x in arrays], axis=axis),
        list=[jnp.concatenate(xs, axis=axis) for xs in zip(*[x.list for x in arrays])],
    )


def stack(arrays: List[e3nn.IrrepsArray], axis=0) -> e3nn.IrrepsArray:
    r"""Stack a list of IrrepsArray.

    Args:
        arrays (list of `IrrepsArray`): list of data to stack
        axis (int): axis to stack on

    Returns:
        `IrrepsArray`: stacked array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0o", jnp.arange(2 * 5).reshape(2, 5))
        >>> y = e3nn.IrrepsArray("3x0e + 2x0o", jnp.arange(2 * 5).reshape(2, 5) + 10)
        >>> e3nn.stack([x, y], axis=0)
        3x0e+2x0o
        [[[ 0  1  2  3  4]
          [ 5  6  7  8  9]]
        <BLANKLINE>
         [[10 11 12 13 14]
          [15 16 17 18 19]]]
        >>> e3nn.stack([x, y], axis=1)
        3x0e+2x0o
        [[[ 0  1  2  3  4]
          [10 11 12 13 14]]
        <BLANKLINE>
         [[ 5  6  7  8  9]
          [15 16 17 18 19]]]
    """
    if len(arrays) == 0:
        raise ValueError("Cannot stack empty list of IrrepsArray")

    result_ndim = arrays[0].ndim + 1
    axis = _standardize_axis(axis, result_ndim)[0]

    jnp = _infer_backend([x.array for x in arrays])

    if axis == result_ndim - 1:
        raise ValueError(
            "IrrepsArray cannot be stacked on the last axis because the last axis is reserved for the irreps dimension"
        )

    if {x.irreps for x in arrays} != {arrays[0].irreps}:
        raise ValueError("Irreps must be the same for all arrays")

    arrays = [
        x.replace_none_with_zeros() for x in arrays
    ]  # TODO this could be optimized
    return e3nn.IrrepsArray(
        irreps=arrays[0].irreps,
        array=jnp.stack([x.array for x in arrays], axis=axis),
        list=[jnp.stack(xs, axis=axis) for xs in zip(*[x.list for x in arrays])],
    )


def norm(
    array: e3nn.IrrepsArray, *, squared: bool = False, per_irrep: bool = True
) -> e3nn.IrrepsArray:
    """Norm of IrrepsArray.

    Args:
        array (IrrepsArray): input array
        squared (bool): if True, return the squared norm
        per_irrep (bool): if True, return the norm of each irrep individually

    Returns:
        IrrepsArray: norm of the input array

    Examples:
        >>> x = e3nn.IrrepsArray("2x0e + 1e + 2e", jnp.arange(10.0))
        >>> e3nn.norm(x)
        2x0e+1x0e+1x0e [ 0.         1.         5.3851647 15.9687195]

        >>> e3nn.norm(x, squared=True)
        2x0e+1x0e+1x0e [  0.   1.  29. 255.]

        >>> e3nn.norm(x, per_irrep=False)
        1x0e [16.881943]
    """
    jnp = _infer_backend(array.array)

    def f(x):
        if x is None:
            return None

        x = jnp.sum(jnp.conj(x) * x, axis=-1, keepdims=True)
        if not squared:
            x_safe = jnp.where(x == 0.0, 1.0, x)
            x_safe = jnp.sqrt(x_safe)
            x = jnp.where(x == 0.0, 0.0, x_safe)
        return x

    if per_irrep:
        return e3nn.IrrepsArray.from_list(
            [(mul, "0e") for mul, _ in array.irreps],
            [f(x) for x in array.list],
            array.shape[:-1],
            array.dtype,
        )
    else:
        return e3nn.IrrepsArray("0e", f(array.array))


def dot(
    a: e3nn.IrrepsArray, b: e3nn.IrrepsArray, per_irrep: bool = False
) -> e3nn.IrrepsArray:
    """Dot product of two IrrepsArray.

    Args:
        a (IrrepsArray): first array (this array get complex conjugated)
        b (IrrepsArray): second array
        per_irrep (bool): if True, return the dot product of each irrep individually

    Returns:
        IrrepsArray: dot product of the two input arrays, as a scalar

    Examples:
        >>> x = e3nn.IrrepsArray("0e + 1e", jnp.array([1.0j, 1.0, 0.0, 0.0]))
        >>> y = e3nn.IrrepsArray("0e + 1e", jnp.array([1.0, 2.0, 1.0, 1.0]))
        >>> e3nn.dot(x, y)
        1x0e [2.-1.j]

        >>> e3nn.dot(x, y, per_irrep=True)
        1x0e+1x0e [0.-1.j 2.+0.j]
    """
    jnp = _infer_backend([a.array, b.array])

    a = a.simplify()
    b = b.simplify()

    if a.irreps != b.irreps:
        raise ValueError(
            "Dot product is only defined for IrrepsArray with the same irreps."
        )

    if per_irrep:
        out = []
        dtype = a.dtype
        for x, y in zip(a.list, b.list):
            if x is None or y is None:
                out.append(None)
            else:
                out.append(jnp.sum(jnp.conj(x) * y, axis=-1, keepdims=True))
                dtype = out[-1].dtype
        return e3nn.IrrepsArray.from_list(
            [(mul, "0e") for mul, _ in a.irreps],
            out,
            a.shape[:-1],
            dtype,
        )
    else:
        out = 0.0
        for x, y in zip(a.list, b.list):
            if x is None or y is None:
                continue
            out = out + jnp.sum(jnp.conj(x) * y, axis=(-2, -1))
        if isinstance(out, float):
            shape = jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1])
            return e3nn.IrrepsArray.zeros("0e", shape, dtype=a.dtype)
        return e3nn.IrrepsArray("0e", out[..., None])


def cross(a: e3nn.IrrepsArray, b: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """Cross product of two IrrepsArray.

    Args:
        a (IrrepsArray): first array of vectors
        b (IrrepsArray): second array of vectors

    Returns:
        IrrepsArray: cross product of the two input arrays

    Examples:
        >>> x = e3nn.IrrepsArray("1o", jnp.array([1.0, 0.0, 0.0]))
        >>> y = e3nn.IrrepsArray("1e", jnp.array([0.0, 1.0, 0.0]))
        >>> e3nn.cross(x, y)
        1x1o [0. 0. 1.]
    """
    jnp = _infer_backend([a.array, b.array])

    if any(ir.l != 1 for _, ir in a.irreps):
        raise ValueError(f"Cross product is only defined for vectors. Got {a.irreps}.")
    if any(ir.l != 1 for _, ir in b.irreps):
        raise ValueError(f"Cross product is only defined for vectors. Got {b.irreps}.")
    if a.irreps.num_irreps != b.irreps.num_irreps:
        raise ValueError(
            "Cross product is only defined for inputs with the same number of vectors."
        )

    a, b = _align_two_irreps_arrays(a, b)
    shape = jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1])

    irreps_out = []
    out = []
    dtype = a.dtype

    for ((mul, irx), x), ((_, iry), y) in zip(
        zip(a.irreps, a.list), zip(b.irreps, b.list)
    ):
        irreps_out.append((mul, (1, irx.p * iry.p)))
        if x is None or y is None:
            out.append(None)
        else:
            out.append(jnp.cross(x, y, axis=-1))
            dtype = out[-1].dtype

    return e3nn.IrrepsArray.from_list(irreps_out, out, shape, dtype)


def normal(
    irreps: IntoIrreps,
    key: jnp.ndarray = None,
    leading_shape: Tuple[int, ...] = (),
    *,
    normalize: bool = False,
    normalization: Optional[str] = None,
    dtype: Optional[jnp.dtype] = None,
) -> e3nn.IrrepsArray:
    r"""Random array with normal distribution.

    Args:
        irreps (Irreps): irreps of the output array
        key (jnp.ndarray): random key (if not provided, use the hash of the irreps as seed, usefull for debugging)
        leading_shape (tuple of int): shape of the leading dimensions
        normalize (bool): if True, normalize the output array
        normalization (str): normalization of the output array, ``"component"`` or ``"norm"``
            This parameter is ignored if ``normalize=False``.
            This parameter only affects the variance distribution.

    Returns:
        IrrepsArray: random array

    Examples:
        >>> jnp.set_printoptions(precision=2, suppress=True)
        >>> e3nn.normal("1o").shape
        (3,)

        Generate a random array with normalization ``"component"``

        >>> x = e3nn.normal("0e + 5e", jax.random.PRNGKey(0), (), normalization="component")
        >>> x
        1x0e+1x5e [ 1.19 -1.1   0.44  0.6  -0.39  0.69  0.46 -2.07 -0.21 -0.99 -0.68  0.27]
        >>> e3nn.norm(x, squared=True)
        1x0e+1x0e [1.42 8.45]

        Generate a random array with normalization ``"norm"``

        >>> x = e3nn.normal("0e + 5e", jax.random.PRNGKey(0), (), normalization="norm")
        >>> x
        1x0e+1x5e [-1.25  0.11 -0.24 -0.4   0.37  0.07  0.15 -0.38  0.35 -0.4   0.03 -0.18]
        >>> e3nn.norm(x, squared=True)
        1x0e+1x0e [1.57 0.85]

        Generate normalized random array

        >>> x = e3nn.normal("0e + 5e", jax.random.PRNGKey(0), (), normalize=True)
        >>> x
        1x0e+1x5e [-1.    0.12 -0.26 -0.43  0.4   0.08  0.16 -0.41  0.37 -0.44  0.03 -0.19]
        >>> e3nn.norm(x, squared=True)
        1x0e+1x0e [1. 1.]
    """
    irreps = e3nn.Irreps(irreps)

    if normalization is None:
        normalization = e3nn.config("irrep_normalization")

    if key is None:
        warnings.warn(
            "e3nn.normal: the key (random seed) is not provided, use the hash of the irreps as key!"
        )
        key = jax.random.PRNGKey(hash(irreps))

    if normalize:
        list = []
        for mul, ir in irreps:
            key, k = jax.random.split(key)
            r = jax.random.normal(k, leading_shape + (mul, ir.dim), dtype=dtype)
            r = r / jnp.linalg.norm(r, axis=-1, keepdims=True)
            list.append(r)
        return e3nn.IrrepsArray.from_list(irreps, list, leading_shape, dtype)
    else:
        if normalization == "component":
            return e3nn.IrrepsArray(
                irreps,
                jax.random.normal(key, leading_shape + (irreps.dim,), dtype=dtype),
            )
        elif normalization == "norm":
            list = []
            for mul, ir in irreps:
                key, k = jax.random.split(key)
                r = jax.random.normal(k, leading_shape + (mul, ir.dim), dtype=dtype)
                r = r / jnp.sqrt(ir.dim)
                list.append(r)
            return e3nn.IrrepsArray.from_list(irreps, list, leading_shape, dtype)
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")
