"""Models with chemistry utilities configuration."""

from typing import Self

import pydantic

from .table import PTABLE


class ChemicalContextConfiguration(pydantic.BaseModel):
    """Store chemical context configuration."""

    custom_abundances: dict[str, dict[int, float]] | None = None
    """Override natural abundances of isotopes. Custom abundances is a dictionary that maps
    atomic numbers to a dictionary of isotopes mass numbers and their abundance. If an abundance
    is not provided for a particular isotope it is assumed to be zero. The sum of abundances for
    a set of isotopologues must be normalized to one."""


class FormulaGeneratorConfiguration(pydantic.BaseModel):
    """Store the formula generator parameters."""

    bounds: dict[str, tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]]
    """A dictionary that maps element (eg: ``"C"``) or isotopes (eg: ``"13C"``) symbols to
    minimum and maximum values of formula coefficients in generated formulas. If element
    symbols are provided, the most abundant isotope is used for formula generation."""

    max_M: pydantic.PositiveFloat
    """Maximum mass value for generated formulas."""

    def update_bounds(self, bounds: dict[str, tuple[int, int]]) -> None:
        """Update or add new bounds."""
        self.bounds.update(bounds)

    @classmethod
    def from_chnops(cls, m: int, **kwargs) -> Self:
        """Create a new instance with predefined bounds for CHNOPS elements.

        CHNOPS bounds were computed by finding the minimum and maximum coefficient bounds for all molecules
        in the `HMDB <http://hmdb.ca>`_ under a specific mass threshold. This function offers precomputed
        bounds using all molecules with mass values under 500, 1000, 1500 and 2000.

        :param m: maximum mass of molecules used to build bounds. Valid values are ``500``, ``1000``,
            ``1500`` or ``2000``.
        :param kwargs: extra arguments passed to the constructor.

        """
        if m == 500:
            bounds = {"C": (0, 34), "H": (0, 70), "N": (0, 10), "O": (0, 18), "P": (0, 4), "S": (0, 7)}
        elif m == 1000:
            bounds = {"C": (0, 70), "H": (0, 128), "N": (0, 15), "O": (0, 31), "P": (0, 8), "S": (0, 7)}
        elif m == 1500:
            bounds = {"C": (0, 100), "H": (0, 164), "N": (0, 23), "O": (0, 46), "P": (0, 8), "S": (0, 7)}
        elif m == 2000:
            bounds = {"C": (0, 108), "H": (0, 190), "N": (0, 23), "O": (0, 61), "P": (0, 8), "S": (0, 8)}
        else:
            raise ValueError(f"Valid mass values are 500, 1000, 1500 or 2000. Got {m}.")

        return cls(bounds=bounds, max_M=m, **kwargs)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_bounds(cls, data):
        for isotope, (lower, upper) in data["bounds"].items():
            _ = PTABLE.get_isotope(isotope)
            if lower >= upper:
                raise ValueError(f"Expected lower bound <= upper bound. Got ({lower}, {upper}) for {isotope}.")
        return data


class EnvelopeScorerConfiguration(FormulaGeneratorConfiguration):
    """Store the envelope scorer configuration."""

    max_length: pydantic.PositiveInt = 5
    """The length of the generated envelopes."""

    context: ChemicalContextConfiguration | None = None
    """The chemical context configuration. If set to ``None`` uses the default chemical context."""


class EnvelopeValidatorConfiguration(EnvelopeScorerConfiguration):
    """Store the envelope validator configuration."""

    min_M_tol: pydantic.PositiveFloat = 0.01
    """Exact mass tolerance for high abundance isotopologues. If ``None``, the parameter is set
    based on the `mode` value. See the notes for an explanation of how this value is used."""

    max_M_tol: pydantic.PositiveFloat = 0.01
    """Exact mass tolerance for low abundance isotopologues.  If ``None``, the parameter is set
    based on the `mode` value. See the notes for an explanation of how this value is used."""

    p_tol: pydantic.PositiveFloat = 0.05
    """Tolerance threshold to include in the abundance results"""

    def remove_elements_with_a_single_isotope(self) -> None:
        """Remove elements with a single isotope from the bounds."""
        self.bounds = {k: v for k, v in self.bounds.items() if len(PTABLE.get_element(k).isotopes) > 1}
