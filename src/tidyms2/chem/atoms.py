"""Tools for working with Isotopes and Elements."""

from __future__ import annotations

from functools import cached_property
from math import isclose
from typing import Any, Sequence

import pydantic

from .exceptions import IsotopeError

EM = 0.00054858  # electron mass


class Isotope(pydantic.BaseModel, frozen=True):
    """Store Isotope mass and abundance information."""

    z: pydantic.PositiveInt
    """The atomic number"""

    a: pydantic.PositiveInt
    """The atomic mass number"""

    m: pydantic.PositiveFloat = pydantic.Field(repr=False)
    """The exact mass"""

    symbol: str
    """The element symbol"""

    p: float = pydantic.Field(ge=0.0, le=1.0, repr=False)
    """The isotope abundance"""

    @pydantic.computed_field(repr=False)
    @cached_property
    def n(self) -> int:
        """The number of neutrons."""
        return self.a - self.z

    @pydantic.computed_field(repr=False)
    @cached_property
    def d(self) -> float:
        """The mass defect."""
        return self.m - self.a

    def to_str(self) -> str:
        """Create a string representation of the isotope."""
        return f"{self.a}{self.symbol}"


class Element(pydantic.BaseModel, frozen=True):
    """Store element information and its isotopes."""

    name: str = pydantic.Field(repr=False)
    """The element name"""

    symbol: str
    """The element symbol"""

    isotopes: Sequence[Isotope] = pydantic.Field(repr=False)
    """Maps mass numbers to isotopes"""

    z: pydantic.PositiveInt
    """The atomic number"""

    @pydantic.field_validator("isotopes")
    @classmethod
    def _validate_isotope_abundances(cls, isotopes: Sequence[Isotope]) -> Sequence[Isotope]:
        total = sum(x.p for x in isotopes)
        if not isclose(total, 1.0):
            z = isotopes[0].z
            raise IsotopeError(
                f"The total isotope abundance of an element must be equal to 1.0. Got {total} for z={z}"
            )
        return isotopes

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_isotopes_z(cls, data: Any) -> Any:
        z = data.get("z")
        assert isinstance(z, int)
        if not all(z == x.z for x in data["isotopes"]):
            raise IsotopeError("All isotopes in an element must have the same atomic number.")
        return data

    @pydantic.computed_field(repr=False)
    @cached_property
    def mmi(self) -> Isotope:
        """Return the isotope with the lowest atomic mass."""
        return min(self.isotopes, key=lambda x: x.a)

    @pydantic.computed_field(repr=False)
    @cached_property
    def monoisotope(self) -> Isotope:
        """Return the most abundant isotope."""
        return max(self.isotopes, key=lambda x: x.p)
