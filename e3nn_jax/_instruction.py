"""Defines an instruction for a tensor product."""

from typing import Tuple
from dataclasses import dataclass, field, replace


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Instruction:
    """Defines an instruction for a tensor product."""

    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    first_input_multiplicity: int
    second_input_multiplicity: int
    output_multiplicity: int
    path_shape: Tuple[int, ...] = field(init=False)
    num_elements: int = field(init=False)

    def __post_init__(self):
        if self.connection_mode not in [
            "uvw",
            "uvu",
            "uvv",
            "uuw",
            "uuu",
            "uvuv",
            "uvu<v",
            "u<vw",
        ]:
            raise ValueError(f"Unsupported connection_mode {self.connection_mode} for instruction.")

        path_shape = {
            "uvw": (
                self.first_input_multiplicity,
                self.second_input_multiplicity,
                self.output_multiplicity,
            ),
            "uvu": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uuw": (self.first_input_multiplicity, self.output_multiplicity),
            "uuu": (self.first_input_multiplicity,),
            "uvuv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvu<v": (self.first_input_multiplicity * (self.second_input_multiplicity - 1) // 2,),
            "u<vw": (
                self.first_input_multiplicity * (self.second_input_multiplicity - 1) // 2,
                self.output_multiplicity,
            ),
        }[self.connection_mode]
        super().__setattr__("path_shape", path_shape)

        num_elements = {
            "uvw": (self.first_input_multiplicity * self.second_input_multiplicity),
            "uvu": self.second_input_multiplicity,
            "uvv": self.first_input_multiplicity,
            "uuw": self.first_input_multiplicity,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.first_input_multiplicity * (self.second_input_multiplicity - 1) // 2,
        }[self.connection_mode]
        super().__setattr__("num_elements", num_elements)

    def replace(self, **changes) -> "Instruction":
        return replace(self, **changes)
