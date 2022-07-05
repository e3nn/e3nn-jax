import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from e3nn_jax import Irreps, axis_angle_to_angles, matrix_to_angles, quaternion_to_angles


class IrrepsData:
    r"""Class storing data and its irreps"""

    irreps: Irreps
    contiguous: jnp.ndarray  # this field is mendatory because it contains the shape
    _list: List[Optional[jnp.ndarray]]  # this field is lazy, it is computed only when needed

    def __init__(self, irreps: Irreps, contiguous: jnp.ndarray, list: List[Optional[jnp.ndarray]] = None):
        self.irreps = Irreps(irreps)
        self.contiguous = contiguous
        self._list = list

    @property
    def list(self) -> List[Optional[jnp.ndarray]]:
        if self._list is None:
            shape = self.contiguous.shape[:-1]
            if len(self.irreps) == 1:
                mul, ir = self.irreps[0]
                list = [jnp.reshape(self.contiguous, shape + (mul, ir.dim))]
            else:
                list = [
                    jnp.reshape(self.contiguous[..., i], shape + (mul, ir.dim))
                    for i, (mul, ir) in zip(self.irreps.slices(), self.irreps)
                ]
            self._list = list
        return self._list

    @staticmethod
    def zeros(irreps: Irreps, shape) -> "IrrepsData":
        irreps = Irreps(irreps)
        return IrrepsData(irreps=irreps, contiguous=jnp.zeros(shape + (irreps.dim,)), list=[None] * len(irreps))

    @staticmethod
    def ones(irreps: Irreps, shape) -> "IrrepsData":
        irreps = Irreps(irreps)
        return IrrepsData(
            irreps=irreps,
            contiguous=jnp.ones(shape + (irreps.dim,)),
            list=[jnp.ones(shape + (mul, ir.dim)) for mul, ir in irreps],
        )

    @staticmethod
    def new(irreps: Irreps, any) -> "IrrepsData":
        r"""Create a new IrrepsData

        Args:
            irreps (`Irreps`): the irreps of the data
            any: the data

        Returns:
            `IrrepsData`
        """
        if isinstance(any, IrrepsData):
            return any.convert(irreps)
        if isinstance(any, list):
            shape = None
            for x in any:
                if x is not None:
                    shape = x.shape[:-2]
            if shape is None:
                raise ValueError("IrrepsData.new cannot infer shape from list of arrays")
            return IrrepsData.from_list(irreps, any, shape)
        return IrrepsData.from_contiguous(irreps, any)

    @staticmethod
    def from_list(irreps: Irreps, list, shape: Tuple[int]) -> "IrrepsData":
        r"""Create an IrrepsData from a list of arrays

        Args:
            irreps (Irreps): irreps
            list (list of optional `jnp.ndarray`): list of arrays

        Returns:
            IrrepsData
        """
        irreps = Irreps(irreps)
        assert len(irreps) == len(list), f"irreps has {len(irreps)} elements and list has {len(list)}"
        assert all(x is None or isinstance(x, jnp.ndarray) for x in list)
        assert all(x is None or x.shape == shape + (mul, ir.dim) for x, (mul, ir) in zip(list, irreps)), (
            [x.shape for x in list if x is not None],
            shape,
        )

        if irreps.dim > 0:
            contiguous = jnp.concatenate(
                [
                    jnp.zeros(shape + (mul_ir.dim,)) if x is None else x.reshape(shape + (mul_ir.dim,))
                    for mul_ir, x in zip(irreps, list)
                ],
                axis=-1,
            )
        else:
            contiguous = jnp.zeros(shape + (0,))
        return IrrepsData(irreps=irreps, contiguous=contiguous, list=list)

    @staticmethod
    def from_contiguous(irreps: Irreps, contiguous) -> "IrrepsData":
        r"""Create an IrrepsData from a contiguous array

        Args:
            irreps (Irreps): irreps
            contiguous (`jnp.ndarray`): contiguous array

        Returns:
            IrrepsData
        """
        assert contiguous.shape[-1] == Irreps(irreps).dim
        return IrrepsData(irreps=irreps, contiguous=contiguous, list=None)

    def __repr__(self):
        return f"IrrepsData({self.irreps}, {self.contiguous}, {self.list})"

    @property
    def shape(self):
        return self.contiguous.shape[:-1]

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, shape) -> "IrrepsData":
        list = [None if x is None else x.reshape(shape + (mul, ir.dim)) for (mul, ir), x in zip(self.irreps, self.list)]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous.reshape(shape + (self.irreps.dim,)), list=list)

    def broadcast_to(self, shape) -> "IrrepsData":
        assert isinstance(shape, tuple)
        contiguous = jnp.broadcast_to(self.contiguous, shape + (self.irreps.dim,))
        list = [
            None if x is None else jnp.broadcast_to(x, shape + (mul, ir.dim)) for (mul, ir), x in zip(self.irreps, self.list)
        ]
        return IrrepsData(irreps=self.irreps, contiguous=contiguous, list=list)

    def replace_none_with_zeros(self) -> "IrrepsData":
        list = [jnp.zeros(self.shape + (mul, ir.dim)) if x is None else x for (mul, ir), x in zip(self.irreps, self.list)]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous, list=list)

    def remove_nones(self) -> "IrrepsData":
        if any(x is None for x in self.list):
            irreps = [mul_ir for mul_ir, x in zip(self.irreps, self.list) if x is not None]
            list = [x for x in self.list if x is not None]
            return IrrepsData.from_list(irreps, list, self.shape)
        return self

    def simplify(self) -> "IrrepsData":
        return self.convert(self.irreps.simplify())

    def split(self, indices: List[int]) -> List["IrrepsData"]:
        contiguous_parts = jnp.split(self.contiguous, [self.irreps[:i].dim for i in indices], axis=-1)
        assert len(contiguous_parts) == len(indices) + 1
        return [
            IrrepsData(irreps=self.irreps[i:j], contiguous=contiguous, list=self.list[i:j])
            for (i, j), contiguous in zip(zip([0] + indices, indices + [len(self.irreps)]), contiguous_parts)
        ]

    def __getitem__(self, index) -> "IrrepsData":
        return IrrepsData(
            self.irreps,
            contiguous=self.contiguous[index],
            list=[None if x is None else x[index] for x in self.list],
        )

    def repeat_irreps_by_last_axis(self) -> "IrrepsData":
        assert len(self.shape) >= 1
        irreps = (self.shape[-1] * self.irreps).simplify()
        contiguous = self.contiguous.reshape(self.shape[:-1] + (irreps.dim,))
        return IrrepsData.from_contiguous(irreps, contiguous)

    def repeat_mul_by_last_axis(self) -> "IrrepsData":
        assert len(self.shape) >= 1
        irreps = Irreps([(self.shape[-1] * mul, ir) for mul, ir in self.irreps])
        list = [None if x is None else x.reshape(self.shape[:-1] + (mul, ir.dim)) for (mul, ir), x in zip(irreps, self.list)]
        return IrrepsData.from_list(irreps, list, self.shape[:-1])

    def factor_irreps_to_last_axis(self) -> "IrrepsData":
        raise NotImplementedError

    def factor_mul_to_last_axis(self, factor=None) -> "IrrepsData":
        if factor is None:
            factor = math.gcd(*(mul for mul, _ in self.irreps))

        if not all(mul % factor == 0 for mul, _ in self.irreps):
            raise ValueError(f"factor {factor} does not divide all multiplicities")

        irreps = Irreps([(mul // factor, ir) for mul, ir in self.irreps])
        list = [
            None if x is None else x.reshape(self.shape + (factor, mul, ir.dim)) for (mul, ir), x in zip(irreps, self.list)
        ]
        return IrrepsData.from_list(irreps, list, self.shape + (factor,))

    def transform_by_angles(self, alpha: float, beta: float, gamma: float, k: int = 0) -> "IrrepsData":
        r"""Rotate the data by angles according to the irreps

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            IrrepsData
        """
        # Optimization: we use only the list of arrays, not the contiguous data
        D = {ir: ir.D_from_angles(alpha, beta, gamma, k) for ir in {ir for _, ir in self.irreps}}
        new_list = [
            jnp.reshape(jnp.einsum("ij,...uj->...ui", D[ir], x), self.shape + (mul, ir.dim)) if x is not None else None
            for (mul, ir), x in zip(self.irreps, self.list)
        ]
        return IrrepsData.from_list(self.irreps, new_list, self.shape)

    def transform_by_quaternion(self, q: jnp.ndarray, k: int = 0) -> "IrrepsData":
        r"""Rotate data by a rotation given by a quaternion

        Args:
            q (`jnp.ndarray`): quaternion
            k (int): parity operation

        Returns:
            IrrepsData
        """
        return self.transform_by_angles(*quaternion_to_angles(q), k)

    def transform_by_axis_angle(self, axis: jnp.ndarray, angle: float, k: int = 0) -> "IrrepsData":
        r"""Rotate data by a rotation given by an axis and an angle

        Args:
            axis (`jnp.ndarray`): axis
            angle (float): angle (in radians)
            k (int): parity operation

        Returns:
            IrrepsData
        """
        return self.transform_by_angles(*axis_angle_to_angles(axis, angle), k)

    def transform_by_matrix(self, R: jnp.ndarray) -> "IrrepsData":
        r"""Rotate data by a rotation given by a matrix

        Args:
            R (`jnp.ndarray`): rotation matrix

        Returns:
            IrrepsData
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.transform_by_angles(*matrix_to_angles(R), k)

    def convert(self, irreps: Irreps) -> "IrrepsData":
        r"""Convert the list property into an equivalent irreps

        Args:
            irreps (Irreps): new irreps

        Returns:
            `IrrepsData`: data with the new irreps

        Raises:
            ValueError: if the irreps are not compatible

        Example:
        >>> id = IrrepsData.new("10x0e + 10x0e", [None, jnp.ones((1, 10, 1))])
        >>> jax.tree_map(lambda x: x.shape, id.convert("20x0e")).list
        [(1, 20, 1)]
        """
        # Optimization: we use only the list of arrays, not the contiguous data
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)
        # TODO test cases with mul == 0

        shape = self.shape

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
                        current_array = jnp.concatenate([current_array, jnp.zeros(shape + (m, mul_ir.ir.dim))], axis=-2)
                else:
                    if isinstance(current_array, int):
                        if current_array == 0:
                            current_array = x
                        else:
                            current_array = jnp.concatenate([jnp.zeros(shape + (current_array, mul_ir.ir.dim)), x], axis=-2)
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
        assert all(x is None or isinstance(x, (jnp.ndarray, np.ndarray)) for x in new_list), [type(x) for x in new_list]
        assert all(x is None or x.shape[-2:] == (mul, ir.dim) for x, (mul, ir) in zip(new_list, irreps))

        return IrrepsData(irreps=irreps, contiguous=self.contiguous, list=new_list)

    def __add__(self, other: "IrrepsData") -> "IrrepsData":
        assert self.irreps == other.irreps
        list = [x if y is None else (y if x is None else x + y) for x, y in zip(self.list, other.list)]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous + other.contiguous, list=list)

    def __sub__(self, other: "IrrepsData") -> "IrrepsData":
        assert self.irreps == other.irreps
        list = [x if y is None else (-y if x is None else x - y) for x, y in zip(self.list, other.list)]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous - other.contiguous, list=list)

    def __mul__(self, other) -> "IrrepsData":
        other = jnp.array(other)
        list = [None if x is None else x * other[..., None, None] for x in self.list]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous * other[..., None], list=list)

    def __rmul__(self, other) -> "IrrepsData":
        return self * other

    def __truediv__(self, other) -> "IrrepsData":
        other = jnp.array(other)
        list = [None if x is None else x / other[..., None, None] for x in self.list]
        return IrrepsData(irreps=self.irreps, contiguous=self.contiguous / other[..., None], list=list)

    @staticmethod
    def cat(args, axis="irreps"):
        r"""Concatenate IrrepsData

        Args:
            args (list of `IrrepsData`): list of data to concatenate
            axis (str or int): axis to concatenate on

        Returns:
            `IrrepsData`: concatenated data
        """
        assert len(args) >= 1
        if axis == "irreps":
            irreps = Irreps(sum([x.irreps for x in args], Irreps("")))
            return IrrepsData(
                irreps=irreps,
                contiguous=jnp.concatenate([x.contiguous for x in args], axis=-1),
                list=sum([x.list for x in args], []),
            )
        elif axis == "mul":
            raise NotImplementedError

        assert isinstance(axis, int)
        assert {x.irreps for x in args} == {args[0].irreps}
        while axis < 0:
            axis += len(args[0].shape)
        args = [x.replace_none_with_zeros() for x in args]  # TODO this could be optimized
        return IrrepsData(
            irreps=args[0].irreps,
            contiguous=jnp.concatenate([x.contiguous for x in args], axis=axis),
            list=[jnp.concatenate(xs, axis=axis) for xs in zip(*[x.list for x in args])],
        )

    @staticmethod
    def randn(irreps, key, size=(), *, normalization=None):
        irreps = Irreps(irreps)
        x = irreps.randn(key, size + (-1,), normalization=normalization)
        return IrrepsData.from_contiguous(irreps, x)


jax.tree_util.register_pytree_node(
    IrrepsData,
    lambda x: ((x.contiguous, x.list), x.irreps),
    lambda x, data: IrrepsData(irreps=x, contiguous=data[0], list=data[1]),
)
