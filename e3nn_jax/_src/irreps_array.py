import functools
import math
import operator
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union
from attr import attrs, attrib

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax._src.irreps import IntoIrreps


def _infer_backend(pytree):
    any_numpy = any(
        isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree)
    )
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


@attrs(frozen=True, repr=False)
class IrrepsArray:
    r"""Array with a representation of rotations.

    The IrrepsArray class enforce equivariance by storing an array of data (``.array``)
    along with its representation (``.irreps``).

    The data is stored as a single array of shape ``(..., irreps.dim)``.

    The data can be accessed as a list of arrays (``.chunks``) matching each item of the ``.irreps``.

    Args:
        irreps (Irreps): representation of the data
        array (`jax.numpy.ndarray`): the data, an array of shape ``(..., irreps.dim)``
        zero_flags (tuple of bool, optional): whether each chunk of the data is zero

    Examples:
        >>> import e3nn_jax as e3nn
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> y = e3nn.from_chunks("1o + 2x0e", [None, jnp.ones((2, 1))], ())
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

    irreps: Irreps = attrib(converter=Irreps)
    array: jnp.ndarray = attrib()
    _zero_flags: Optional[Tuple[bool, ...]] = attrib(
        default=None, kw_only=True, converter=lambda x: None if x is None else tuple(x)
    )

    def __post_init__(self):
        if hasattr(self.array, "shape"):
            if self.array.shape[-1] != self.irreps.dim:
                raise ValueError(
                    f"IrrepsArray: Array shape {self.array.shape} incompatible with irreps {self.irreps}. "
                    f"{self.array.shape[-1]} != {self.irreps.dim}"
                )
        if self.zero_flags is not None:
            if len(self.zero_flags) != len(self.irreps):
                raise ValueError(
                    f"IrrepsArray: len(zero_flags) != len(irreps), {len(self.zero_flags)} != {len(self.irreps)}"
                )

    @staticmethod
    def from_list(
        irreps: IntoIrreps,
        chunks: List[Optional[jnp.ndarray]],
        leading_shape: Tuple[int, ...],
        dtype=None,
        *,
        backend=None,
    ):
        warnings.warn(
            "IrrepsArray.from_list is deprecated, use e3nn.from_chunks instead.",
            DeprecationWarning,
        )
        return e3nn.from_chunks(irreps, chunks, leading_shape, dtype, backend=backend)

    @staticmethod
    def as_irreps_array(array: Union[jnp.ndarray, "IrrepsArray"], *, backend=None):
        warnings.warn(
            "IrrepsArray.as_irreps_array is deprecated, use e3nn.as_irreps_array instead.",
            DeprecationWarning,
        )
        return e3nn.as_irreps_array(array)

    @staticmethod
    def zeros(irreps: IntoIrreps, leading_shape, dtype=None) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.zeros is deprecated, use e3nn.zeros instead.",
            DeprecationWarning,
        )
        return e3nn.zeros(irreps, leading_shape, dtype)

    @staticmethod
    def zeros_like(irreps_array: "IrrepsArray") -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.zeros_like is deprecated, use e3nn.zeros_like instead.",
            DeprecationWarning,
        )
        return e3nn.zeros_like(irreps_array)

    @property
    def list(self) -> List[Optional[jnp.ndarray]]:
        warnings.warn(
            "IrrepsArray.list is deprecated, use IrrepsArray.chunks instead.",
            DeprecationWarning,
        )
        return self.chunks

    @property
    def chunks(self) -> List[Optional[jnp.ndarray]]:
        r"""List of arrays matching each item of the ``.irreps``.

        Examples:
            >>> x = IrrepsArray("2x0e + 0e", jnp.arange(3))
            >>> len(x.chunks)
            2
            >>> x.chunks[0]
            Array([[0],
                   [1]], dtype=int32)
            >>> x.chunks[1]
            Array([[2]], dtype=int32)

            The follwing is always true:

            >>> all(e.shape == x.shape[:-1] + (mul, ir.dim) for (mul, ir), e in zip(x.irreps, x.chunks))
            True
        """
        jnp = _infer_backend(self.array)
        leading_shape = self.array.shape[:-1]
        if self.zero_flags is None:
            zeros = [False] * len(self.irreps)
        else:
            zeros = self.zero_flags

        if len(self.irreps) == 1:
            mul, ir = self.irreps[0]
            if zeros[0]:
                return [None]
            return [jnp.reshape(self.array, leading_shape + (mul, ir.dim))]
        else:
            return [
                None
                if zero
                else jnp.reshape(self.array[..., i], leading_shape + (mul, ir.dim))
                for zero, i, (mul, ir) in zip(zeros, self.irreps.slices(), self.irreps)
            ]

    @property
    def zero_flags(self):
        if self._zero_flags is None:
            return (False,) * len(self.irreps)
        return self._zero_flags

    @property
    def shape(self):
        r"""Shape. Equivalent to ``self.array.shape``."""
        return self.array.shape

    @property
    def dtype(self):
        r"""dtype. Equivalent to ``self.array.dtype``."""
        return self.array.dtype

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
        r = str(self.array)
        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"

    def __len__(self):  # noqa: D105
        return len(self.array)

    def __eq__(
        self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

        if isinstance(other, IrrepsArray):
            if self.irreps != other.irreps:
                raise ValueError(
                    "IrrepsArray({self.irreps}) == IrrepsArray({other.irreps}) is not equivariant."
                )

            leading_shape = jnp.broadcast_shapes(self.shape[:-1], other.shape[:-1])

            def eq(mul: int, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                if x is None and y is None:
                    return jnp.ones(leading_shape + (mul,), bool)
                if x is None:
                    x = 0.0
                if y is None:
                    y = 0.0

                return jnp.all(x == y, axis=-1)

            chunks = [
                eq(mul, x, y)[..., None]
                for (mul, ir), x, y in zip(self.irreps, self.chunks, other.chunks)
            ]
            return e3nn.from_chunks(
                [(mul, "0e") for mul, _ in self.irreps], chunks, leading_shape, bool
            )

        other = jnp.asarray(other)
        if self.irreps.lmax > 0 or (other.ndim > 0 and other.shape[-1] != 1):
            raise ValueError(
                f"IrrepsArray({self.irreps}) == scalar(shape={other.shape}) is not equivariant."
            )
        return IrrepsArray(self.irreps, self.array == other)

    def __neg__(self: "IrrepsArray") -> "IrrepsArray":
        return IrrepsArray(self.irreps, -self.array, zero_flags=self.zero_flags)

    def __add__(
        self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray, float, int]
    ) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        jnp = _infer_backend(self.array)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(self.irreps, self.array + other)
            raise ValueError(f"IrrepsArray({self.irreps}) + scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}) + IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        return IrrepsArray(self.irreps, self.array + other.array, zero_flags=zero_flags)

    def __radd__(
        self: "IrrepsArray", other: Union[jnp.ndarray, float, int]
    ) -> "IrrepsArray":
        return self + other

    def __sub__(
        self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray, float, int]
    ) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        jnp = _infer_backend(self.array)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(irreps=self.irreps, array=self.array - other)
            raise ValueError(f"IrrepsArray({self.irreps}) - scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}) - IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        return IrrepsArray(self.irreps, self.array - other.array, zero_flags=zero_flags)

    def __rsub__(
        self: "IrrepsArray", other: Union[jnp.ndarray, float, int]
    ) -> "IrrepsArray":
        return -self + other

    def __mul__(
        self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

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
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", other)
            return e3nn.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}) * scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(self.irreps, self.array * other, zero_flags=self.zero_flags)

    def __rmul__(
        self: "IrrepsArray", other: jnp.ndarray
    ) -> "IrrepsArray":  # noqa: D105
        return self * other

    def __truediv__(
        self: "IrrepsArray", other: Union["IrrepsArray", jnp.ndarray]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

        if isinstance(other, IrrepsArray):
            if (
                len(other.irreps) == 0
                or other.irreps.lmax > 0
                or self.irreps.num_irreps != other.irreps.num_irreps
            ):
                raise ValueError(
                    f"IrrepsArray({self.irreps}) / IrrepsArray({other.irreps}) is not equivariant."
                )

            if any(x is None for x in other.chunks):
                raise ValueError(
                    "There are deterministic Zeros in the array of the lhs. Cannot divide by Zero."
                )
            other = 1.0 / other
            return e3nn.elementwise_tensor_product(self, other)

        other = jnp.asarray(other)
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", 1.0 / other)
            return e3nn.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}) / scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(self.irreps, self.array / other, zero_flags=self.zero_flags)

    def __rtruediv__(
        self: "IrrepsArray", other: jnp.ndarray
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend((self.array, other))

        other = jnp.asarray(other)
        if self.irreps.lmax > 0:
            raise ValueError(
                f"scalar(shape={other.shape}) / IrrepsArray({self.irreps}) is not equivariant."
            )
        if any(x is None for x in self.chunks):
            raise ValueError(
                "There are deterministic Zeros in the array of the lhs. Cannot divide by Zero."
            )

        return IrrepsArray(self.irreps, other / self.array)

    def __pow__(self, exponent) -> "IrrepsArray":  # noqa: D105
        if all(ir == "0e" for _, ir in self.irreps):
            return IrrepsArray(self.irreps, self.array**exponent)

        if exponent % 1.0 == 0.0 and self.irreps.lmax == 0:
            irreps = self.irreps
            if exponent % 2.0 == 0.0:
                irreps = [(mul, "0e") for mul, ir in self.irreps]
            return IrrepsArray(irreps, array=self.array**exponent)

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

            ii = [
                i
                for i in range(len(self.irreps))
                if self.irreps[i : i + len(irreps)] == irreps
            ]
            if len(ii) != 1:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Can't slice with {irreps} "
                    f"because it doesn't appear exactly once in {self.irreps}."
                )
            i = ii[0]

            return IrrepsArray(
                irreps,
                self.array[
                    ..., self.irreps[:i].dim : self.irreps[: i + len(irreps)].dim
                ],
                zero_flags=self.zero_flags[i : i + len(irreps)],
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
                        return self.rechunk(
                            self.irreps[: i - 1]
                            + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)])
                            + self.irreps[i:]
                        )[index]

                if self.irreps[:i].dim == stop:
                    irreps_stop = i
                    break

                if irreps_stop is None and stop < self.irreps[:i].dim:
                    # "2x1e"[:3]
                    mul, ir = self.irreps[i - 1]
                    if (stop - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (stop - self.irreps[: i - 1].dim) // ir.dim
                        return self.rechunk(
                            self.irreps[: i - 1]
                            + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)])
                            + self.irreps[i:]
                        )[index]

            if irreps_start is None or irreps_stop is None:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, unable to slice {self.irreps} with {start}:{stop}."
                )

            return IrrepsArray(
                self.irreps[irreps_start:irreps_stop],
                self.array[..., start:stop],
                zero_flags=self.zero_flags[irreps_start:irreps_stop],
            )[index[:-1] + (slice(None),)]

        # Prevent None at last index  x[..., None] and x[:, :, None]
        if (
            len(index[:-1]) == self.ndim or any(map(_is_ellipse, index[:-1]))
        ) and index[-1] is None:
            raise IndexError(
                "Error in IrrepsArray.__getitem__, cannot add a new dimension at the end."
            )

        # Prevent indexing the last axis
        if (len(index) == self.ndim or any(map(_is_ellipse, index[:-1]))) and not (
            _is_ellipse(index[-1]) or _is_none_slice(index[-1]) or index[-1] is None
        ):
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
        return IrrepsArray(self.irreps, self.array[index], zero_flags=self.zero_flags)

    @property
    def at(self):
        return _IndexUpdateHelper(self)

    def reshape(self, shape) -> "IrrepsArray":
        r"""Reshape the array.

        Args:
            shape (tuple): new shape

        Returns:
            IrrepsArray: new IrrepsArray

        Examples:
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
        return IrrepsArray(
            self.irreps,
            self.array.reshape(shape[:-1] + (self.irreps.dim,)),
            zero_flags=self.zero_flags,
        )

    def astype(self, dtype) -> "IrrepsArray":
        r"""Change the dtype of the array.

        Args:
            dtype (dtype): new dtype

        Returns:
            IrrepsArray: new IrrepsArray
        """
        return IrrepsArray(
            irreps=self.irreps,
            array=self.array.astype(dtype),
            zero_flags=self.zero_flags,
        )

    def remove_nones(self) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.remove_nones is deprecated. Use IrrepsArray.remove_zero_chunks instead.",
            DeprecationWarning,
        )
        return self.remove_zero_chunks()

    def remove_zero_chunks(self) -> "IrrepsArray":
        r"""Remove all zero chunks."""
        irreps = Irreps(
            [mul_ir for mul_ir, zero in zip(self.irreps, self.zero_flags) if not zero]
        )
        chunks = [x for x, zero in zip(self.chunks, self.zero_flags) if not zero]
        return e3nn.from_chunks(
            irreps,
            chunks,
            self.shape[:-1],
            self.dtype,
            backend=_infer_backend(self.array),
        )

    def simplify(self) -> "IrrepsArray":
        r"""Simplify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0e + 0e", jnp.ones(3)).simplify()
            3x0e [1. 1. 1.]

            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).simplify()
            2x0e [1. 1.]
        """
        return self.rechunk(self.irreps.simplify())

    def unify(self) -> "IrrepsArray":
        r"""Unify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).unify()
            1x0e+0x1e+1x0e [1. 1.]
        """
        return self.rechunk(self.irreps.unify())

    def sort(self) -> "IrrepsArray":
        r"""Sort the irreps.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).sort()
            1x0e+2x0e+1x1o [0 4 5 1 2 3]
        """
        irreps, p, inv = self.irreps.sort()
        return e3nn.from_chunks(
            irreps,
            [self.chunks[i] for i in inv],
            self.shape[:-1],
            self.dtype,
            backend=_infer_backend(self.array),
        )

    def sorted(self) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.sorted is deprecated, use IrrepsArray.sort instead.",
            DeprecationWarning,
        )
        return self.sort()

    def regroup(self) -> "IrrepsArray":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sorted` followed by :meth:`simplify`.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).regroup()
            3x0e+1x1o [0 4 5 1 2 3]
        """
        return self.sort().simplify()

    def filter(
        self,
        keep: Union[
            e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]
        ] = None,
        *,
        drop: Union[
            e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]
        ] = None,
        lmax: int = None,
    ) -> "IrrepsArray":
        r"""Filter the irreps.

        Args:
            keep (Irreps or list of `Irrep` or function): list of irrep to keep
            exclude (Irreps or list of `Irrep` or function): list of irrep to exclude
            lmax (int): maximum l

        Examples:
            >>> IrrepsArray("0e + 2x1o + 2x0e", jnp.arange(9)).filter(["1o"])
            2x1o [1 2 3 4 5 6]
        """
        if keep is None and drop is None and lmax is None:
            return self

        backend = _infer_backend(self.array)
        new_irreps = self.irreps.filter(keep=keep, drop=drop, lmax=lmax)
        return e3nn.from_chunks(
            new_irreps,
            [x for x, mul_ir in zip(self.chunks, self.irreps) if mul_ir in new_irreps],
            self.shape[:-1],
            self.dtype,
            backend=backend,
        )

    def filtered(self, *args, **kwargs) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.filtered is deprecated, use IrrepsArray.filter instead.",
            DeprecationWarning,
        )
        return self.filter(*args, **kwargs)

    @property
    def slice_by_mul(self):
        r"""Return the slice with respect to the multiplicities.

        See also:
            :meth:`Irreps.slice_by_mul`
        """
        return _MulIndexSliceHelper(self)

    @property
    def slice_by_dim(self):
        r"""Same as ``__getitem__`` in the irreps dimension.

        See also:
            :meth:`Irreps.slice_by_dim`
        """
        return _DimIndexSliceHelper(self)

    @property
    def slice_by_chunk(self):
        r"""Return the slice with respect to the chunks.

        See also:
            :meth:`Irreps.slice_by_chunk`
        """
        return _ChunkIndexSliceHelper(self)

    def axis_to_irreps(self, axis: int = -2) -> "IrrepsArray":
        r"""Repeat the irreps by the last axis of the array.

        Examples:
            >>> x = IrrepsArray("0e + 1e", jnp.arange(2 * 4).reshape(2, 4))
            >>> x.axis_to_irreps()
            1x0e+1x1e+1x0e+1x1e [0 1 2 3 4 5 6 7]
        """
        assert self.ndim >= 2
        axis = _standardize_axis(axis, self.ndim)[0]
        jnp = _infer_backend(self.array)

        new_irreps = self.irreps.repeat(self.shape[axis]).simplify()
        new_array = jnp.moveaxis(self.array, axis, -2)
        new_array = jnp.reshape(new_array, self.shape[:-2] + (new_irreps.dim,))
        return IrrepsArray(new_irreps, new_array)

    repeat_irreps_by_last_axis = axis_to_irreps

    def irreps_to_axis(self) -> "IrrepsArray":  # noqa: D102
        raise NotImplementedError

    # Move multiplicity to the previous last axis and back

    def mul_to_axis(
        self, factor: Optional[int] = None, axis: int = -2
    ) -> "IrrepsArray":
        r"""Create a new axis in the previous last position by factoring the multiplicities.

        Increase the dimension of the array by 1.

        Args:
            factor (int or None): factor the multiplicities by this number
            axis (int): the new axis will be placed before this axis

        Examples:
            >>> x = IrrepsArray("6x0e + 3x1e", jnp.arange(15))
            >>> x.mul_to_axis()
            2x0e+1x1e
            [[ 0  1  6  7  8]
             [ 2  3  9 10 11]
             [ 4  5 12 13 14]]
        """
        axis = _standardize_axis(axis, self.ndim + 1)
        if axis == self.ndim:
            raise ValueError(
                "axis cannot be the last axis. The last axis is reserved for the irreps dimension."
            )

        if factor is None:
            factor = functools.reduce(math.gcd, (mul for mul, _ in self.irreps))

        if not all(mul % factor == 0 for mul, _ in self.irreps):
            raise ValueError(
                f"factor {factor} does not divide all multiplicities: {self.irreps}"
            )

        irreps = Irreps([(mul // factor, ir) for mul, ir in self.irreps])
        new_list = [
            None if x is None else x.reshape(self.shape[:-1] + (factor, mul, ir.dim))
            for (mul, ir), x in zip(irreps, self.chunks)
        ]
        new_list = [None if x is None else jnp.moveaxis(x, -3, axis) for x in new_list]
        return e3nn.from_chunks(
            irreps, new_list, self.shape[:-1] + (factor,), self.dtype
        )

    def factor_mul_to_last_axis(self, axis: int = -2) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.factor_mul_to_last_axis is deprecated. Use IrrepsArray.mul_to_axis instead.",
            DeprecationWarning,
        )
        return self.mul_to_axis(axis=axis)

    def axis_to_mul(self, axis: int = -2) -> "IrrepsArray":
        r"""Repeat the multiplicity by the previous last axis of the array.

        Decrease the dimension of the array by 1.

        Args:
            axis (int): axis to convert into multiplicity

        Examples:
            >>> x = IrrepsArray("0e + 1e", jnp.arange(2 * 4).reshape(2, 4))
            >>> x.axis_to_mul()
            2x0e+2x1e [0 4 1 2 3 5 6 7]
        """
        assert self.ndim >= 2
        axis = _standardize_axis(axis, self.ndim)[0]

        if axis == self.ndim - 1:
            raise ValueError(
                "The last axis is the irreps dimension and therefore cannot be converted to multiplicity."
            )

        new_list = [
            None if x is None else jnp.moveaxis(x, axis, -3) for x in self.chunks
        ]
        new_irreps = Irreps([(self.shape[-2] * mul, ir) for mul, ir in self.irreps])
        new_list = [
            None if x is None else x.reshape(self.shape[:-2] + (new_mul, ir.dim))
            for (new_mul, ir), x in zip(new_irreps, new_list)
        ]
        return e3nn.from_chunks(new_irreps, new_list, self.shape[:-2], self.dtype)

    def repeat_mul_by_last_axis(self, axis: int = -2) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.repeat_mul_by_last_axis is deprecated. Use IrrepsArray.axis_to_mul instead.",
            DeprecationWarning,
        )
        return self.axis_to_mul(axis=axis)

    def transform_by_log_coordinates(
        self, log_coordinates: jnp.ndarray, k: int = 0
    ) -> "IrrepsArray":
        r"""Rotate data by a rotation given by log coordinates.

        Args:
            log_coordinates (`jax.numpy.ndarray`): log coordinates
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data
        """
        log_coordinates = log_coordinates.astype(self.dtype)
        D = {
            ir: ir.D_from_log_coordinates(log_coordinates, k)
            for ir in {ir for _, ir in self.irreps}
        }
        new_list = [
            jnp.reshape(
                jnp.einsum("ij,...uj->...ui", D[ir], x), self.shape[:-1] + (mul, ir.dim)
            )
            if x is not None
            else None
            for (mul, ir), x in zip(self.irreps, self.chunks)
        ]
        return e3nn.from_chunks(self.irreps, new_list, self.shape[:-1], self.dtype)

    def transform_by_angles(
        self, alpha: float, beta: float, gamma: float, k: int = 0, inverse: bool = False
    ) -> "IrrepsArray":
        r"""Rotate the data by angles according to the irreps.

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation
            inverse (bool): if True, apply the inverse rotation

        Returns:
            `IrrepsArray`: rotated data

        Examples:
            >>> np.set_printoptions(precision=3, suppress=True)
            >>> x = IrrepsArray("2e", jnp.array([0.1, 2, 1.0, 1, 1]))
            >>> x.transform_by_angles(jnp.pi, 0, 0)
            1x2e [ 0.1 -2.   1.  -1.   1. ]
        """
        alpha = (
            alpha
            if isinstance(alpha, (int, float))
            else jnp.asarray(alpha, dtype=self.dtype)
        )
        beta = (
            beta
            if isinstance(beta, (int, float))
            else jnp.asarray(beta, dtype=self.dtype)
        )
        gamma = (
            gamma
            if isinstance(gamma, (int, float))
            else jnp.asarray(gamma, dtype=self.dtype)
        )
        D = {
            ir: ir.D_from_angles(alpha, beta, gamma, k)
            for ir in {ir for _, ir in self.irreps}
        }
        if inverse:
            D = {ir: jnp.swapaxes(D[ir], -2, -1) for ir in D}
        new_list = [
            jnp.reshape(
                jnp.einsum("ij,...uj->...ui", D[ir], x), self.shape[:-1] + (mul, ir.dim)
            )
            if x is not None
            else None
            for (mul, ir), x in zip(self.irreps, self.chunks)
        ]
        return e3nn.from_chunks(self.irreps, new_list, self.shape[:-1], self.dtype)

    def transform_by_quaternion(self, q: jnp.ndarray, k: int = 0) -> "IrrepsArray":
        r"""Rotate data by a rotation given by a quaternion.

        Args:
            q (`jax.numpy.ndarray`): quaternion
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data
        """
        return self.transform_by_log_coordinates(
            e3nn.quaternion_to_log_coordinates(q), k
        )

    def transform_by_axis_angle(
        self, axis: jnp.ndarray, angle: float, k: int = 0
    ) -> "IrrepsArray":
        r"""Rotate data by a rotation given by an axis and an angle.

        Args:
            axis (`jax.numpy.ndarray`): axis
            angle (float): angle (in radians)
            k (int): parity operation

        Returns:
            `IrrepsArray`: rotated data
        """
        return self.transform_by_log_coordinates(
            e3nn.axis_angle_to_log_coordinates(axis, angle), k
        )

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
        return self.transform_by_angles(*e3nn.matrix_to_angles(R), k)

    def rechunk(self, irreps: IntoIrreps) -> "IrrepsArray":
        r"""Rechunk the array with new (equivalent) irreps.

        Args:
            irreps (Irreps): new irreps

        Returns:
            `IrrepsArray`: new IrrepsArray

        Examples:
            >>> x = e3nn.from_chunks("6x0e + 4x0e", [None, jnp.ones((4, 1))], ())
            >>> x.rechunk("5x0e + 5x0e").chunks
            [None, Array([[0.],
                   [1.],
                   [1.],
                   [1.],
                   [1.]], dtype=float32)]
        """
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)

        if len(self.irreps) == 0:
            zero_flags = []
        else:
            zero_flags = np.concatenate(
                [
                    z * np.ones(mul * ir.dim, dtype=bool)
                    for z, (mul, ir) in zip(self.zero_flags, self.irreps)
                ]
            )
            zero_flags = [bool(np.all(zero_flags[s])) for s in irreps.slices()]

        return IrrepsArray(irreps, self.array, zero_flags=zero_flags)

    def broadcast_to(self, shape) -> "IrrepsArray":
        """Broadcast the array to a new shape."""
        jnp = _infer_backend(self.array)

        assert isinstance(shape, tuple)
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        leading_shape = shape[:-1]
        array = jnp.broadcast_to(self.array, leading_shape + (self.irreps.dim,))
        return IrrepsArray(self.irreps, array, zero_flags=self.zero_flags)


# We purposefully do not register zero_flags
jax.tree_util.register_pytree_node(
    IrrepsArray,
    lambda x: ((x.array,), x.irreps),
    lambda irreps, data: IrrepsArray(irreps, data[0]),
)


def _standardize_axis(
    axis: Union[None, int, Tuple[int, ...]], result_ndim: int
) -> Tuple[int, ...]:
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
                raise IndexError(
                    f"Indexing with {index[-1]} in the irreps dimension is not supported."
                )

        # Support of x.at[index, :].set(0)
        if isinstance(values, (int, float)) and values == 0:
            return IrrepsArray(
                self.irreps,
                array=self.array.at[index].set(0),
                zero_flags=self.zero_flags,
            )

        # Support of x.at[index, :].set(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError(
                    "The irreps of the array and the values to set must be the same."
                )

            values = values.rechunk(self.irreps)

            zero_flags = tuple(
                x and y for x, y in zip(self.zero_flags, values.zero_flags)
            )
            return IrrepsArray(
                self.irreps,
                self.array.at[index].set(values.array),
                zero_flags=zero_flags,
            )

        raise NotImplementedError(
            f"x.at[i].set(v) with v={type(values)} is not implemented."
        )

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
                raise IndexError(
                    f"Indexing with {index[-1]} in the irreps dimension is not supported."
                )

        # Support of x.at[index, :].add(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError(
                    "The irreps of the array and the values to add must be the same."
                )

            values = values.rechunk(self.irreps)

            zero_flags = tuple(
                x and y for x, y in zip(self.zero_flags, values.zero_flags)
            )
            return IrrepsArray(
                self.irreps,
                self.array.at[index].add(values.array),
                zero_flags=zero_flags,
            )

        raise NotImplementedError(
            f"x.at[i].add(v) with v={type(values)} is not implemented."
        )


class _MulIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_mul only supports one slices (like IrrepsArray.slice_by_mul[2:4])."
            )
        start, stop, stride = index.indices(self.irreps_array.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError(
                "IrrepsArray.slice_by_mul does not support strides."
            )

        irreps = []
        list = []
        i = 0
        for (mul, ir), x in zip(self.irreps_array.irreps, self.irreps_array.chunks):
            if start <= i and i + mul <= stop:
                irreps.append((mul, ir))
                list.append(x)
            elif start < i + mul and i < stop:
                irreps.append((min(stop, i + mul) - max(start, i), ir))
                list.append(x[..., max(start, i) - i : min(stop, i + mul) - i, :])

            i += mul
        return e3nn.from_chunks(
            irreps,
            list,
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
            backend=_infer_backend(self.irreps_array.array),
        )


class _DimIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_dim only supports slices (like IrrepsArray.slice_by_dim[2:4])."
            )
        return self.irreps_array[..., index]


class _ChunkIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_chunk only supports slices (like IrrepsArray.slice_by_chunk[2:4])."
            )
        start, stop, stride = index.indices(len(self.irreps_array.irreps))

        return e3nn.from_chunks(
            self.irreps_array.irreps[start:stop:stride],
            self.irreps_array.chunks[start:stop:stride],
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
            backend=_infer_backend(self.irreps_array.array),
        )
