import functools
import math
import operator
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import Irreps, axis_angle_to_angles, config, matrix_to_angles, quaternion_to_angles
from e3nn_jax._src.irreps import IntoIrreps


def _infer_backend(pytree):
    any_numpy = any(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree))
    any_jax = any(isinstance(x, jnp.ndarray) for x in jax.tree_util.tree_leaves(pytree))
    if any_numpy and any_jax:
        raise ValueError("Cannot mix numpy and jax arrays")
    if any_numpy:
        return np
    if any_jax:
        return jnp
    return jnp


def _is_ellipse(x):
    return type(x) == type(Ellipsis)


def _is_none_slice(x):
    return isinstance(x, slice) and x == slice(None)


class IrrepsArray:
    r"""Data along with its irreps.

    The IrrepsArray class enforce equivariance by storing an array of data (``.array``)
    along with its representation (``.irreps``).

    Args:
        irreps (Irreps): representation of the data
        array (`jax.numpy.ndarray`): the data, an array of shape ``(..., irreps.dim)``
        list (list of jax.numpy.ndarray or None, optional): the same data in a list format.
            It can contain ``None`` to represent zeros otherwise the shape has to be ``(..., mul, ir.dim)``.

    Examples:
        >>> import e3nn_jax as e3nn
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> y = e3nn.IrrepsArray.from_list("1o + 2x0e", [None, jnp.ones((2, 1))], ())
        >>> x + y
        1x1o+2x0e [1. 1. 1. 2. 2.]

        Example of indexing:

        >>> x = IrrepsArray("0e + 1o", jnp.arange(2 * 4).reshape(2, 4))
        >>> x[0]
        1x0e+1x1o [0 1 2 3]
        >>> x[1, "0e"]
        1x0e [4]
        >>> x[:, 1:]
        1x1o
        [[1 2 3]
         [5 6 7]]
        >>> IrrepsArray("5x0e", jnp.arange(5))[1:3]
        2x0e [1 2]
    """

    irreps: Irreps
    list: List[Optional[jnp.ndarray]]
    _aval: jnp.ndarray  # dummy array of shape [..., mul=0, ir]

    def __init__(self, irreps: IntoIrreps, array: jnp.ndarray, list=None, *, _internal_init=None):
        """Create an IrrepsArray."""
        self.irreps = Irreps(irreps)

        if list is not None:
            warnings.warn("IrrepsArray(irreps, array, list) is deprecated", DeprecationWarning)
            _internal_init = (list, array[..., :0][..., None])

        if _internal_init is not None:
            self.list = _internal_init[0]
            self._aval = _internal_init[1]
        else:
            if array.shape[-1] != self.irreps.dim:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, {array.shape}) the last axis of the array "
                    "is expected to match the dimension of the irreps."
                )

            jnp = _infer_backend(array)

            leading_shape = array.shape[:-1]
            if len(self.irreps) == 1:
                mul, ir = self.irreps[0]
                list = [jnp.reshape(array, leading_shape + (mul, ir.dim))]
            else:
                list = [
                    jnp.reshape(array[..., i], leading_shape + (mul, ir.dim))
                    for i, (mul, ir) in zip(self.irreps.slices(), self.irreps)
                ]
            self.list = list
            self._aval = array[..., :0][..., None]  # [*leading_shape, 0, 1]

    @staticmethod
    def from_list(
        irreps: IntoIrreps,
        list: List[Optional[jnp.ndarray]],
        leading_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[jnp.dtype] = None,
        *,
        _aval=None,
    ) -> "IrrepsArray":
        r"""Create an IrrepsArray from a list of arrays.

        Args:
            irreps (Irreps): irreps
            list (list of optional `jax.numpy.ndarray`): list of arrays
            leading_shape (optional, tuple of int): leading shape of the arrays (without the irreps)
            dtype (optional): the data type

        Returns:
            IrrepsArray
        """
        jnp = _infer_backend(list)

        irreps = Irreps(irreps)
        if len(irreps) != len(list):
            raise ValueError(f"IrrepsArray.from_list: len(irreps) != len(list), {len(irreps)} != {len(list)}")

        if not all(x is None or isinstance(x, jnp.ndarray) for x in list):
            raise ValueError(f"IrrepsArray.from_list: list contains non-array elements type={[type(x) for x in list]}")

        if _aval is not None:
            leading_shape = _aval.shape[:-2]
            dtype = _aval.dtype

        if leading_shape is None:
            if all(x is None for x in list):
                raise ValueError(
                    "IrrepsArray.from_list: Cannot obtain the shape from the list because all elements are None, "
                    "please provide leading_shape argument."
                )
            leading_shape = next(x.shape[:-2] for x in list if x is not None)

        if not all(x is None or x.shape == leading_shape + (mul, ir.dim) for x, (mul, ir) in zip(list, irreps)):
            raise ValueError(
                f"IrrepsArray.from_list: list shapes {[None if x is None else x.shape for x in list]} "
                f"incompatible with leading shape {leading_shape} and irreps {irreps}. "
                f"Expecting {[leading_shape + (mul, ir.dim) for (mul, ir) in irreps]}."
            )

        if dtype is None:
            if all(x is None for x in list):
                dtype = np.float32
            else:
                dtype = next(x.dtype for x in list if x is not None)

        if not all(x is None or x.dtype == dtype for x in list):
            raise ValueError(
                f"IrrepsArray.from_list: list dtype {[None if x is None else x.dtype for x in list]} "
                f"are expected to be {dtype}."
            )

        if _aval is None:
            _aval = jnp.empty(leading_shape + (0, 1), dtype)
        return IrrepsArray(irreps=irreps, array=None, _internal_init=(list, _aval))

    @staticmethod
    def zeros(irreps: IntoIrreps, leading_shape, dtype=np.float32) -> "IrrepsArray":
        r"""Create an IrrepsArray of zeros."""
        irreps = Irreps(irreps)
        return IrrepsArray.from_list(irreps, [None] * len(irreps), leading_shape, dtype)

    @staticmethod
    def zeros_like(irreps_array: "IrrepsArray") -> "IrrepsArray":
        r"""Create an IrrepsArray of zeros with the same shape as another IrrepsArray."""
        return IrrepsArray.zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype)

    @staticmethod
    def ones(irreps: IntoIrreps, leading_shape, dtype=np.float32) -> "IrrepsArray":
        r"""Create an IrrepsArray of ones."""
        # TODO: maybe remove this function because it is not equivariant
        irreps = Irreps(irreps)
        return IrrepsArray(irreps, jnp.ones(leading_shape + (irreps.dim,), dtype))

    @property
    def array(self):
        jnp = _infer_backend(self._aval)
        leading_shape = self._aval.shape[:-2]
        return jnp.concatenate(
            [jnp.reshape(self._aval, leading_shape + (0,))]
            + [
                jnp.zeros(leading_shape + (mul * ir.dim,), dtype=self._aval.dtype)
                if x is None
                else jnp.reshape(x, leading_shape + (mul * ir.dim,))
                for (mul, ir), x in zip(self.irreps, self.list)
            ],
            axis=-1,
        )

    @property
    def shape(self):
        r"""Shape. Equivalent to ``self.array.shape``."""
        return self._aval.shape[:-2] + (self.irreps.dim,)

    @property
    def dtype(self):
        r"""dtype. Equivalent to ``self.array.dtype``."""
        return self._aval.dtype

    @property
    def ndim(self):
        r"""Number of dimensions. Equivalent to ``self.array.ndim``."""
        return len(self.shape)

    # def __jax_array__(self):
    #     if self.irreps.lmax > 0:
    #         return NotImplemented
    #     return self.array
    #
    # Note: - __jax_array__ seems to be incompatible with register_pytree_node
    #       - __jax_array__ cause problem for the multiplication: jnp.array * IrrepsArray -> jnp.array

    def __repr__(self):  # noqa: D105
        try:
            r = str(self.array)
        except AttributeError:
            r = str(self.list)

        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"

    def __len__(self):  # noqa: D105
        return self.shape[0]

    def __eq__(self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self._aval)

        if isinstance(other, IrrepsArray):
            if self.irreps != other.irreps:
                raise ValueError("IrrepsArray({self.irreps}) == IrrepsArray({other.irreps}) is not equivariant.")

            leading_shape = jnp.broadcast_shapes(self.shape[:-1], other.shape[:-1])

            def eq(mul: int, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                if x is None and y is None:
                    return jnp.ones(leading_shape + (mul,), dtype="bool")
                if x is None:
                    x = 0.0
                if y is None:
                    y = 0.0

                return jnp.all(x == y, axis=-1)

            list = [eq(mul, x, y)[..., None] for (mul, ir), x, y in zip(self.irreps, self.list, other.list)]
            return IrrepsArray.from_list([(mul, "0e") for mul, _ in self.irreps], list, leading_shape)

        other = jnp.asarray(other)
        if not self.irreps.is_scalar() or (other.ndim > 0 and other.shape[-1] != 1):
            raise ValueError(f"IrrepsArray({self.irreps}) == scalar(shape={other.shape}) is not equivariant.")
        return IrrepsArray(irreps=self.irreps, array=self.array == other)

    def __neg__(self: "IrrepsArray") -> "IrrepsArray":
        return IrrepsArray.from_list(self.irreps, [-x if x is not None else None for x in self.list], _aval=self._aval)

    def __add__(self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self._aval)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(irreps=self.irreps, array=self.array + other)
            raise ValueError(f"IrrepsArray({self.irreps}) + scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(f"IrrepsArray({self.irreps}) + IrrepsArray({other.irreps}) is not equivariant.")

        list = [x if y is None else (y if x is None else x + y) for x, y in zip(self.list, other.list)]
        return IrrepsArray.from_list(self.irreps, list, _aval=self._aval + other._aval)

    def __sub__(self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self._aval)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(irreps=self.irreps, array=self.array - other)
            raise ValueError(f"IrrepsArray({self.irreps}) - scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(f"IrrepsArray({self.irreps}) - IrrepsArray({other.irreps}) is not equivariant.")
        list = [x if y is None else (-y if x is None else x - y) for x, y in zip(self.list, other.list)]
        return IrrepsArray.from_list(self.irreps, list=list, _aval=self._aval - other._aval)

    def __mul__(self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self._aval)

        if isinstance(other, IrrepsArray):
            if self.irreps.num_irreps != other.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}) * IrrepsArray({other.irreps}) only works if the number of irreps is the same."
                )
            irreps_out = e3nn.elementwise_tensor_product(self.irreps, other.irreps)
            if irreps_out.num_irreps != self.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}) * IrrepsArray({other.irreps}) "
                    "is only supported for scalar * irreps and irreps * scalar. "
                    "To perform irreps * irreps use e3nn.elementwise_tensor_product or e3nn.tensor_product."
                )
            return e3nn.elementwise_tensor_product(self, other)

        other = jnp.asarray(other)
        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(f"IrrepsArray({self.irreps}) * scalar(shape={other.shape}) is not equivariant.")
        if other.ndim == 0:
            other = other[None]
        list = [None if x is None else x * other[..., None] for x in self.list]
        return IrrepsArray.from_list(self.irreps, list, _aval=self._aval * other[..., :0][..., None])

    def __rmul__(self: "IrrepsArray", other: jnp.ndarray) -> "IrrepsArray":  # noqa: D105
        return self * other

    def __truediv__(self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self._aval)

        if isinstance(other, IrrepsArray):
            if len(other.irreps) == 0 or other.irreps.lmax > 0 or self.irreps.num_irreps != other.irreps.num_irreps:
                raise ValueError(f"IrrepsArray({self.irreps}) / IrrepsArray({other.irreps}) is not equivariant.")

            if any(x is None for x in other.list):
                raise ValueError("There are deterministic Zeros in the array of the lhs. Cannot divide by Zero.")
            other = 1.0 / other
            return e3nn.elementwise_tensor_product(self, other)

        other = jnp.asarray(other)
        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(f"IrrepsArray({self.irreps}) / scalar(shape={other.shape}) is not equivariant.")
        list = [None if x is None else x / other[..., None] for x in self.list]
        return IrrepsArray(irreps=self.irreps, array=self.array / other, list=list)

    def __rtruediv__(self: "IrrepsArray", other: jnp.ndarray) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend((self.array, other))

        other = jnp.asarray(other)
        if self.irreps.lmax > 0:
            raise ValueError(f"scalar(shape={other.shape}) / IrrepsArray({self.irreps}) is not equivariant.")
        if any(x is None for x in self.list):
            raise ValueError("There are deterministic Zeros in the array of the lhs. Cannot divide by Zero.")

        return IrrepsArray(irreps=self.irreps, array=other / self.array, list=[other[..., None] / x for x in self.list])

    def __pow__(self, exponent) -> "IrrepsArray":  # noqa: D105
        if all(ir == "0e" for _, ir in self.irreps):
            return IrrepsArray(irreps=self.irreps, array=self.array**exponent, list=[x**exponent for x in self.list])

        if exponent % 1.0 == 0.0 and self.irreps.lmax == 0:
            irreps = self.irreps
            if exponent % 2.0 == 0.0:
                irreps = [(mul, "0e") for mul, ir in self.irreps]
            return IrrepsArray(irreps, array=self.array**exponent, list=[x**exponent for x in self.list])

        raise ValueError(f"IrrepsArray({self.irreps}) ** scalar is not equivariant.")

    def __iter__(self):  # noqa: D105
        if self.ndim <= 1:
            raise ValueError("Can't iterate over IrrepsArray with ndim <= 1")
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index) -> "IrrepsArray":  # noqa: D105
        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (e3nn.Irrep, e3nn.MulIrrep, Irreps, str)):
            if not (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim):
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Irreps index must be the last index, try x[..., {index[-1]}]."
                )

            irreps = Irreps(index[-1])

            ii = [i for i in range(len(self.irreps)) if self.irreps[i : i + len(irreps)] == irreps]
            if len(ii) != 1:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Can't slice with {irreps} "
                    f"because it doesn't appear exactly once in {self.irreps}."
                )
            i = ii[0]

            return IrrepsArray(
                irreps,
                self.array[..., self.irreps[:i].dim : self.irreps[: i + len(irreps)].dim],
                self.list[i : i + len(irreps)],
            )[index[:-1] + (slice(None),)]

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and index[-1].step is None
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            start, stop, _ = index[-1].indices(self.shape[-1])

            irreps_start = None
            irreps_stop = None

            for i in range(len(self.irreps) + 1):
                if self.irreps[:i].dim == start:
                    irreps_start = i

                if irreps_start is None and start < self.irreps[:i].dim:
                    # "2x1e"[3:]
                    mul, ir = self.irreps[i - 1]
                    if (start - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (start - self.irreps[: i - 1].dim) // ir.dim
                        return self._convert(
                            self.irreps[: i - 1] + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)]) + self.irreps[i:]
                        )[index]

                if self.irreps[:i].dim == stop:
                    irreps_stop = i
                    break

                if irreps_stop is None and stop < self.irreps[:i].dim:
                    # "2x1e"[:3]
                    mul, ir = self.irreps[i - 1]
                    if (stop - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (stop - self.irreps[: i - 1].dim) // ir.dim
                        return self._convert(
                            self.irreps[: i - 1] + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)]) + self.irreps[i:]
                        )[index]

            if irreps_start is None or irreps_stop is None:
                raise IndexError(f"Error in IrrepsArray.__getitem__, unable to slice {self.irreps} with {start}:{stop}.")

            return IrrepsArray(
                self.irreps[irreps_start:irreps_stop], self.array[..., start:stop], self.list[irreps_start:irreps_stop]
            )[index[:-1] + (slice(None),)]

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                if isinstance(index[-1], int):
                    raise IndexError(
                        f"Error in IrrepsArray.__getitem__, integer index in the irreps dimension is not supported, "
                        f"try x[..., {index[-1]}:{index[-1] + 1}] instead."
                    )
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, indexing the irreps dimension with [..., {index[-1]}] "
                    "is not supported."
                )

        # Support of x[index, :]
        return IrrepsArray(
            self.irreps,
            array=self.array[index],
            list=[None if x is None else x[index + (slice(None),)] for x in self.list],
        )

    @property
    def at(self):
        return _IndexUpdateHelper(self)

    def reshape(self, shape) -> "IrrepsArray":
        r"""Reshape the array.

        Args:
            shape (tuple): new shape

        Returns:
            IrrepsArray: new IrrepsArray

        Example:
            >>> IrrepsArray("2x0e + 1o", jnp.ones((6, 5))).reshape((2, 3, 5))
            2x0e+1x1o
            [[[1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]]
            <BLANKLINE>
             [[1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]]]
        """
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        shape = shape[:-1]
        list = [None if x is None else x.reshape(shape + (mul, ir.dim)) for (mul, ir), x in zip(self.irreps, self.list)]
        return IrrepsArray(irreps=self.irreps, array=self.array.reshape(shape + (self.irreps.dim,)), list=list)

    def replace_none_with_zeros(self) -> "IrrepsArray":
        r"""Replace all None in ``.list`` with zeros."""
        jnp = _infer_backend(self._aval)

        list = [jnp.zeros(self.shape[:-1] + (mul, ir.dim)) if x is None else x for (mul, ir), x in zip(self.irreps, self.list)]
        return IrrepsArray(irreps=self.irreps, array=self.array, list=list)

    def remove_nones(self) -> "IrrepsArray":
        r"""Remove all None in ``.list`` and ``.irreps``."""
        if any(x is None for x in self.list):
            irreps = [mul_ir for mul_ir, x in zip(self.irreps, self.list) if x is not None]
            list = [x for x in self.list if x is not None]
            return IrrepsArray.from_list(irreps, list, self.shape[:-1])
        return self

    def simplify(self) -> "IrrepsArray":
        r"""Simplify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0e + 0e", jnp.ones(3)).simplify()
            3x0e [1. 1. 1.]

            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).simplify()
            2x0e [1. 1.]
        """
        return self._convert(self.irreps.simplify())

    def unify(self) -> "IrrepsArray":
        r"""Unify the irreps.

        Example:
            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).unify()
            1x0e+0x1e+1x0e [1. 1.]
        """
        return self._convert(self.irreps.unify())

    def sort(self) -> "IrrepsArray":
        r"""Sort the irreps.

        Example:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).sorted()
            1x0e+2x0e+1x1o [0 4 5 1 2 3]
        """
        irreps, p, inv = self.irreps.sort()
        return IrrepsArray.from_list(irreps, [self.list[i] for i in inv], self.shape[:-1])

    sorted = sort

    def regroup(self) -> "IrrepsArray":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sorted` followed by :meth:`simplify`.

        Example:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).regroup()
            3x0e+1x1o [0 4 5 1 2 3]
        """
        return self.sorted().simplify()

    def filter(
        self,
        keep: Union[e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]] = None,
        *,
        drop: Union[e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]] = None,
    ) -> "IrrepsArray":
        r"""Filter the irreps.

        Args:
            keep (Irreps or list of `Irrep` or function): list of irrep to keep
            exclude (Irreps or list of `Irrep` or function): list of irrep to exclude

        Example:
            >>> IrrepsArray("0e + 2x1o + 2x0e", jnp.arange(9)).filtered(["1o"])
            2x1o [1 2 3 4 5 6]
        """
        if keep is None and drop is None:
            return self
        if keep is not None and drop is not None:
            raise ValueError("Cannot specify both keep and drop")

        new_irreps = self.irreps.filter(keep=keep, drop=drop)
        return IrrepsArray.from_list(
            new_irreps, [x for x, mul_ir in zip(self.list, self.irreps) if mul_ir in new_irreps], self.shape[:-1]
        )

    filtered = filter

    @property
    def slice_by_mul(self):
        r"""Return the slice with respect to the multiplicities.

        See also:
            :meth:`e3nn.Irreps.slice_by_mul`
        """
        return _MulIndexSliceHelper(self)

    @property
    def slice_by_dim(self):
        r"""Same as ``__getitem__`` in the irreps dimension.

        See also:
            :meth:`e3nn.Irreps.slice_by_dim`
        """
        return _DimIndexSliceHelper(self)

    @property
    def slice_by_chunk(self):
        r"""Return the slice with respect to the chunks.

        See also:
            :meth:`e3nn.Irreps.slice_by_chunk`
        """
        return _ChunkIndexSliceHelper(self)

    def axis_to_irreps(self, axis: int = -2) -> "IrrepsArray":
        r"""Repeat the irreps by the last axis of the array.

        Example:
            >>> x = IrrepsArray("0e + 1e", jnp.arange(2 * 4).reshape(2, 4))
            >>> x.axis_to_irreps()
            1x0e+1x1e+1x0e+1x1e [0 1 2 3 4 5 6 7]
        """
        assert self.ndim >= 2
        axis = _standardize_axis(axis, self.ndim)[0]
        jnp = _infer_backend(self._aval)

        new_irreps = self.irreps.repeat(self.shape[axis]).simplify()
        new_array = jnp.moveaxis(self.array, axis, -2)
        new_array = jnp.reshape(new_array, self.shape[:-2] + (new_irreps.dim,))
        return IrrepsArray(new_irreps, new_array)

    repeat_irreps_by_last_axis = axis_to_irreps

    def irreps_to_axis(self) -> "IrrepsArray":  # noqa: D102
        raise NotImplementedError

    # Move multiplicity to the previous last axis and back

    def mul_to_axis(self, factor: Optional[int] = None, axis: int = -2) -> "IrrepsArray":
        r"""Create a new axis in the previous last position by factoring the multiplicities.

        Increase the dimension of the array by 1.

        Args:
            factor (int or None): factor the multiplicities by this number
            axis (int): the new axis will be placed before this axis

        Example:
            >>> x = IrrepsArray("6x0e + 3x1e", jnp.arange(15))
            >>> x.mul_to_axis()
            2x0e+1x1e
            [[ 0  1  6  7  8]
             [ 2  3  9 10 11]
             [ 4  5 12 13 14]]
        """
        axis = _standardize_axis(axis, self.ndim + 1)
        if axis == self.ndim:
            raise ValueError("axis cannot be the last axis. The last axis is reserved for the irreps dimension.")

        if factor is None:
            factor = functools.reduce(math.gcd, (mul for mul, _ in self.irreps))

        if not all(mul % factor == 0 for mul, _ in self.irreps):
            raise ValueError(f"factor {factor} does not divide all multiplicities: {self.irreps}")

        irreps = Irreps([(mul // factor, ir) for mul, ir in self.irreps])
        new_list = [
            None if x is None else x.reshape(self.shape[:-1] + (factor, mul, ir.dim))
            for (mul, ir), x in zip(irreps, self.list)
        ]
        new_list = [None if x is None else jnp.moveaxis(x, -3, axis) for x in new_list]
        return IrrepsArray.from_list(irreps, new_list, self.shape[:-1] + (factor,))

    def axis_to_mul(self, axis: int = -2) -> "IrrepsArray":
        r"""Repeat the multiplicity by the previous last axis of the array.

        Decrease the dimension of the array by 1.

        Args:
            axis (int): axis to convert into multiplicity

        Example:
            >>> x = IrrepsArray("0e + 1e", jnp.arange(2 * 4).reshape(2, 4))
            >>> x.axis_to_mul()
            2x0e+2x1e [0 4 1 2 3 5 6 7]
        """
        assert self.ndim >= 2
        axis = _standardize_axis(axis, self.ndim)[0]

        if axis == self.ndim - 1:
            raise ValueError("The last axis is the irreps dimension and therefore cannot be converted to multiplicity.")

        new_list = [None if x is None else jnp.moveaxis(x, axis, -3) for x in self.list]
        new_irreps = Irreps([(self.shape[-2] * mul, ir) for mul, ir in self.irreps])
        new_list = [
            None if x is None else x.reshape(self.shape[:-2] + (new_mul, ir.dim))
            for (new_mul, ir), x in zip(new_irreps, new_list)
        ]
        return IrrepsArray.from_list(new_irreps, new_list, self.shape[:-2])

    repeat_mul_by_last_axis = axis_to_mul
    factor_mul_to_last_axis = mul_to_axis

    def transform_by_angles(self, alpha: float, beta: float, gamma: float, k: int = 0) -> "IrrepsArray":
        r"""Rotate the data by angles according to the irreps.

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data

        Example:
            >>> np.set_printoptions(precision=3, suppress=True)
            >>> x = IrrepsArray("2e", jnp.array([0.1, 0, 1.0, 1, 1]))
            >>> x.transform_by_angles(jnp.pi, 0, 0)
            1x2e [ 0.1  0.   1.  -1.   1. ]
        """
        # Optimization: we use only the list of arrays, not the array data
        D = {ir: ir.D_from_angles(alpha, beta, gamma, k) for ir in {ir for _, ir in self.irreps}}
        new_list = [
            jnp.reshape(jnp.einsum("ij,...uj->...ui", D[ir], x), self.shape[:-1] + (mul, ir.dim)) if x is not None else None
            for (mul, ir), x in zip(self.irreps, self.list)
        ]
        return IrrepsArray.from_list(self.irreps, new_list, self.shape[:-1])

    def transform_by_quaternion(self, q: jnp.ndarray, k: int = 0) -> "IrrepsArray":
        r"""Rotate data by a rotation given by a quaternion.

        Args:
            q (`jax.numpy.ndarray`): quaternion
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data
        """
        return self.transform_by_angles(*quaternion_to_angles(q), k)

    def transform_by_axis_angle(self, axis: jnp.ndarray, angle: float, k: int = 0) -> "IrrepsArray":
        r"""Rotate data by a rotation given by an axis and an angle.

        Args:
            axis (`jax.numpy.ndarray`): axis
            angle (float): angle (in radians)
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data
        """
        return self.transform_by_angles(*axis_angle_to_angles(axis, angle), k)

    def transform_by_matrix(self, R: jnp.ndarray) -> "IrrepsArray":
        r"""Rotate data by a rotation given by a matrix.

        Args:
            R (`jax.numpy.ndarray`): rotation matrix

        Returns:
            `IrrepsArray`: rotated data
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.transform_by_angles(*matrix_to_angles(R), k)

    def _convert(self, irreps: IntoIrreps) -> "IrrepsArray":
        r"""Convert the list property into an equivalent irreps.

        Args:
            irreps (Irreps): new irreps

        Returns:
            `IrrepsArray`: data with the new irreps

        Raises:
            ValueError: if the irreps are not compatible

        Example:
            >>> x = IrrepsArray.from_list("6x0e + 4x0e", [None, jnp.ones((4, 1))], ())
            >>> x._convert("5x0e + 5x0e").list
            [None, DeviceArray([[0.],
                         [1.],
                         [1.],
                         [1.],
                         [1.]], dtype=float32)]
        """
        jnp = _infer_backend(self._aval)

        # Optimization: we use only the list of arrays, not the array data
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)
        # TODO test cases with mul == 0

        leading_shape = self.shape[:-1]

        new_list = []
        current_array = 0

        while len(new_list) < len(irreps) and irreps[len(new_list)].mul == 0:
            new_list.append(None)

        for mul_ir, y in zip(self.irreps, self.list):
            mul, _ = mul_ir

            while mul > 0:
                if isinstance(current_array, int):
                    current_mul = current_array
                else:
                    current_mul = current_array.shape[-2]

                needed_mul = irreps[len(new_list)].mul - current_mul

                if mul <= needed_mul:
                    x = y
                    m = mul
                    mul = 0
                elif mul > needed_mul:
                    if y is None:
                        x = None
                    else:
                        x, y = jnp.split(y, [needed_mul], axis=-2)
                    m = needed_mul
                    mul -= needed_mul

                if x is None:
                    if isinstance(current_array, int):
                        current_array += m
                    else:
                        current_array = jnp.concatenate(
                            [current_array, jnp.zeros(leading_shape + (m, mul_ir.ir.dim))], axis=-2
                        )
                else:
                    if isinstance(current_array, int):
                        if current_array == 0:
                            current_array = x
                        else:
                            current_array = jnp.concatenate(
                                [jnp.zeros(leading_shape + (current_array, mul_ir.ir.dim)), x], axis=-2
                            )
                    else:
                        current_array = jnp.concatenate([current_array, x], axis=-2)

                if isinstance(current_array, int):
                    if current_array == irreps[len(new_list)].mul:
                        new_list.append(None)
                        current_array = 0
                else:
                    if current_array.shape[-2] == irreps[len(new_list)].mul:
                        new_list.append(current_array)
                        current_array = 0

                while len(new_list) < len(irreps) and irreps[len(new_list)].mul == 0:
                    new_list.append(None)

        assert current_array == 0

        assert len(new_list) == len(irreps)
        assert all(x is None or isinstance(x, jnp.ndarray) for x in new_list), [type(x) for x in new_list]
        assert all(x is None or x.shape[-2:] == (mul, ir.dim) for x, (mul, ir) in zip(new_list, irreps))

        return IrrepsArray(irreps=irreps, array=self.array, list=new_list)

    def broadcast_to(self, shape) -> "IrrepsArray":
        """Broadcast the array to a new shape."""
        jnp = _infer_backend(self._aval)

        assert isinstance(shape, tuple)
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        leading_shape = shape[:-1]
        array = jnp.broadcast_to(self.array, leading_shape + (self.irreps.dim,))
        list = [
            None if x is None else jnp.broadcast_to(x, leading_shape + (mul, ir.dim))
            for (mul, ir), x in zip(self.irreps, self.list)
        ]
        return IrrepsArray(irreps=self.irreps, array=array, list=list)

    @staticmethod
    def cat(args, axis=-1) -> "IrrepsArray":  # noqa: D102
        warnings.warn("IrrepsArray.cat is deprecated, use e3nn.concatenate instead", DeprecationWarning)
        return concatenate(args, axis=axis)

    @staticmethod
    def randn(irreps, key, leading_shape=(), *, normalization=None):  # noqa: D102
        warnings.warn("IrrepsArray.randn is deprecated, use e3nn.normal instead", DeprecationWarning)
        return normal(irreps, key, leading_shape=leading_shape, normalization=normalization)


jax.tree_util.register_pytree_node(
    IrrepsArray,
    lambda x: ((x.list, x._aval), x.irreps),
    lambda extra, leaves: IrrepsArray(extra, None, _internal_init=leaves),
)


def _standardize_axis(axis: Union[None, int, Tuple[int, ...]], result_ndim: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(result_ndim))
    try:
        axis = (operator.index(axis),)
    except TypeError:
        axis = tuple(operator.index(i) for i in axis)

    if not all(-result_ndim <= i < result_ndim for i in axis):
        raise ValueError("axis out of range")
    axis = tuple(i % result_ndim for i in axis)

    return tuple(sorted(set(axis)))


def _reduce(op, array: IrrepsArray, axis: Union[None, int, Tuple[int, ...]] = None, keepdims: bool = False) -> IrrepsArray:
    axis = _standardize_axis(axis, array.ndim)

    if axis == ():
        return array

    if axis[-1] < array.ndim - 1:
        # irrep dimension is not affected by mean
        return IrrepsArray(
            array.irreps,
            op(array.array, axis=axis, keepdims=keepdims),
            [None if x is None else op(x, axis=axis, keepdims=keepdims) for x in array.list],
        )

    array = _reduce(op, array, axis=axis[:-1], keepdims=keepdims)
    return IrrepsArray.from_list(
        Irreps([(1, ir) for _, ir in array.irreps]),
        [None if x is None else op(x, axis=-2, keepdims=True) for x in array.list],
        array.shape[:-1],
    )


def mean(array: IrrepsArray, axis: Union[None, int, Tuple[int, ...]] = None, keepdims: bool = False) -> IrrepsArray:
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


def sum_(array: IrrepsArray, axis: Union[None, int, Tuple[int, ...]] = None, keepdims: bool = False) -> IrrepsArray:
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
    """
    jnp = _infer_backend(array.array)
    return _reduce(jnp.sum, array, axis, keepdims)


def concatenate(arrays: List[IrrepsArray], axis: int = -1) -> IrrepsArray:
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

    axis = _standardize_axis(axis, arrays[0].ndim)[0]

    jnp = _infer_backend([x.array for x in arrays])

    if axis == arrays[0].ndim - 1:
        irreps = Irreps(sum([x.irreps for x in arrays], Irreps("")))
        return IrrepsArray(
            irreps=irreps,
            array=jnp.concatenate([x.array for x in arrays], axis=-1),
            list=sum([x.list for x in arrays], []),
        )

    if {x.irreps for x in arrays} != {arrays[0].irreps}:
        raise ValueError("Irreps must be the same for all arrays")

    arrays = [x.replace_none_with_zeros() for x in arrays]  # TODO this could be optimized
    return IrrepsArray(
        irreps=arrays[0].irreps,
        array=jnp.concatenate([x.array for x in arrays], axis=axis),
        list=[jnp.concatenate(xs, axis=axis) for xs in zip(*[x.list for x in arrays])],
    )


def stack(arrays: List[IrrepsArray], axis=0) -> IrrepsArray:
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

    arrays = [x.replace_none_with_zeros() for x in arrays]  # TODO this could be optimized
    return IrrepsArray(
        irreps=arrays[0].irreps,
        array=jnp.stack([x.array for x in arrays], axis=axis),
        list=[jnp.stack(xs, axis=axis) for xs in zip(*[x.list for x in arrays])],
    )


def norm(array: IrrepsArray, *, squared: bool = False) -> IrrepsArray:
    """Norm of IrrepsArray.

    Args:
        array (IrrepsArray): input array
        squared (bool): if True, return the squared norm

    Returns:
        IrrepsArray: norm of the input array

    Example:
        >>> x = e3nn.IrrepsArray("2x0e + 1e + 2e", jnp.arange(10))
        >>> e3nn.norm(x)
        2x0e+1x0e+1x0e [ 0.     1.     5.385 15.969]
    """
    jnp = _infer_backend(array.array)

    def f(x):
        x = jnp.sum(x**2, axis=-1, keepdims=True)
        if not squared:
            x = jnp.sqrt(x)
        return x

    return IrrepsArray.from_list(
        [(mul, "0e") for mul, _ in array.irreps],
        [f(x) for x in array.list],
        array.shape[:-1],
    )


def normal(
    irreps: IntoIrreps,
    key: jnp.ndarray = None,
    leading_shape: Tuple[int, ...] = (),
    *,
    normalize: bool = False,
    normalization: Optional[str] = None,
) -> IrrepsArray:
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
        >>> np.set_printoptions(precision=2, suppress=True)
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
    irreps = Irreps(irreps)

    if normalization is None:
        normalization = config("irrep_normalization")

    if key is None:
        warnings.warn("e3nn.normal: the key (random seed) is not provided, use the hash of the irreps as key!")
        key = jax.random.PRNGKey(hash(irreps))

    if normalize:
        list = []
        for mul, ir in irreps:
            key, k = jax.random.split(key)
            r = jax.random.normal(k, leading_shape + (mul, ir.dim))
            r = r / jnp.linalg.norm(r, axis=-1, keepdims=True)
            list.append(r)
        return IrrepsArray.from_list(irreps, list, leading_shape)
    else:
        if normalization == "component":
            return IrrepsArray(irreps, jax.random.normal(key, leading_shape + (irreps.dim,)))
        elif normalization == "norm":
            list = []
            for mul, ir in irreps:
                key, k = jax.random.split(key)
                r = jax.random.normal(k, leading_shape + (mul, ir.dim))
                r = r / jnp.sqrt(ir.dim)
                list.append(r)
            return IrrepsArray.from_list(irreps, list, leading_shape)
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")


class _IndexUpdateHelper:
    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.irreps_array, index)


class _IndexUpdateRef:
    def __init__(self, irreps_array, index) -> None:
        self.irreps_array = irreps_array
        self.index = index

    def set(self, values: Any) -> IrrepsArray:
        index = self.index
        self = self.irreps_array

        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (e3nn.Irrep, e3nn.MulIrrep, Irreps, str)):
            raise NotImplementedError('x.at[..., "1e + 2e"] is not implemented')

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and index[-1].step is None
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            raise NotImplementedError("x.at[..., 3:32] is not implemented")

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                raise IndexError(f"Indexing with {index[-1]} in the irreps dimension is not supported.")

        # Support of x.at[index, :].set(0)
        if isinstance(values, (int, float)) and values == 0:
            return IrrepsArray(
                self.irreps,
                array=self.array.at[index].set(0),
                list=[None if x is None else x.at[index + (slice(None),)].set(0) for x in self.list],
            )

        # Support of x.at[index, :].set(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError("The irreps of the array and the values to set must be the same.")

            values = values._convert(self.irreps)

            def fn(x, y, mul, ir):
                if x is not None and y is not None:
                    return x.at[index + (slice(None),)].set(y)
                if x is not None and y is None:
                    return x.at[index + (slice(None),)].set(0)
                if x is None and y is not None:
                    return jnp.zeros(self.shape[:-1] + (mul, ir.dim), dtype=self.array.dtype).at[index + (slice(None),)].set(y)
                if x is None and y is None:
                    return None

            return IrrepsArray(
                self.irreps,
                array=self.array.at[index].set(values.array),
                list=[fn(x, y, mul, ir) for (mul, ir), x, y in zip(self.irreps, self.list, values.list)],
            )

        raise NotImplementedError(f"x.add[i].set(v) with v={type(values)} is not implemented.")

    def add(self, values: Any) -> IrrepsArray:
        index = self.index
        self = self.irreps_array

        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (e3nn.Irrep, e3nn.MulIrrep, Irreps, str)):
            raise NotImplementedError('x.at[..., "1e + 2e"] is not implemented')

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and index[-1].step is None
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            raise NotImplementedError("x.at[..., 3:32] is not implemented")

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                raise IndexError(f"Indexing with {index[-1]} in the irreps dimension is not supported.")

        # Support of x.at[index, :].add(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError("The irreps of the array and the values to add must be the same.")

            values = values._convert(self.irreps)

            def fn(x, y, mul, ir):
                if x is not None and y is not None:
                    return x.at[index + (slice(None),)].add(y)
                if x is not None and y is None:
                    return x
                if x is None and y is not None:
                    return jnp.zeros(self.shape[:-1] + (mul, ir.dim), dtype=self.array.dtype).at[index + (slice(None),)].add(y)
                if x is None and y is None:
                    return None

            return IrrepsArray(
                self.irreps,
                array=self.array.at[index].add(values.array),
                list=[fn(x, y, mul, ir) for (mul, ir), x, y in zip(self.irreps, self.list, values.list)],
            )

        raise NotImplementedError(f"x.add[i].add(v) with v={type(values)} is not implemented.")


class _MulIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_mul only supports slices.")
        start, stop, stride = index.indices(self.irreps_array.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("IrrepsArray.slice_by_mul does not support strides.")

        irreps = []
        list = []
        i = 0
        for (mul, ir), x in zip(self.irreps_array.irreps, self.irreps_array.list):
            if start <= i and i + mul <= stop:
                irreps.append((mul, ir))
                list.append(x)
            elif start < i + mul and i < stop:
                irreps.append((min(stop, i + mul) - max(start, i), ir))
                list.append(x[..., max(start, i) - i : min(stop, i + mul) - i, :])

            i += mul
        return IrrepsArray.from_list(irreps, list, self.irreps_array.shape[:-1])


class _DimIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_dim only supports slices.")
        return self.irreps_array[..., index]


class _ChunkIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_chunk only supports slices.")
        start, stop, stride = index.indices(len(self.irreps_array.irreps))

        return IrrepsArray.from_list(
            self.irreps_array.irreps[start:stop:stride],
            self.irreps_array.list[start:stop:stride],
            self.irreps_array.shape[:-1],
        )
