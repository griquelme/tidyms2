"""Utilities to compute isotopic envelopes."""

from __future__ import annotations

from functools import cache
from typing import Generator, Sequence

import numpy as np
import pydantic
from scipy.stats import multinomial

from ..core.utils.numpy import FloatArray, FloatArray1D, IntArray1D, cartesian_product_from_iterable
from .atoms import Element, Isotope


class Bounds(pydantic.BaseModel):
    """Store lower and upper bounds for mass."""

    lower: float
    upper: float

    @pydantic.model_validator(mode="after")
    def _validate_bounds(self):
        if self.lower >= self.upper:
            raise ValueError("Lower bounds must be lower than the upper bound.")
        return self


class EnvelopeBounds(pydantic.BaseModel):
    """Store exact mass and abundance bounds for an envelope query."""

    M: Bounds
    """Exact mass bounds."""

    p: Bounds
    """Abundance bounds."""

    index: int
    """The index used to generate the bounds."""


class EnvelopeQueryTolerance(pydantic.BaseModel):
    """Store the mass and abundance tolerance."""

    min_M_tol: pydantic.PositiveFloat

    max_M_tol: pydantic.PositiveFloat

    p_tol: float = pydantic.Field(lt=1.0, gt=0.0)

    @pydantic.model_validator(mode="after")
    def _check_mass_tolerance(self):
        if self.min_M_tol > self.max_M_tol:
            raise ValueError("`min_M_tol` must be lower than `max_M_tol`.")
        return self


class EnvelopeQuery(pydantic.BaseModel):
    """Container class for Envelope Queries.

    :param M: exact mass of the envelope
    :param p: abundance of the envelope.

    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    M: Sequence[float]
    p: Sequence[float]

    @pydantic.field_validator("M", mode="before")
    @classmethod
    def _check_sorted_mass(cls, M: list[float]) -> list[float]:
        if sorted(M) != M:
            raise ValueError("Values in M must be sorted.")
        return M

    @pydantic.field_validator("M", "p", mode="before")
    @classmethod
    def _check_all_gt_zero(cls, v: list[float]) -> list[float]:
        if any(x <= 0.0 for x in v):
            raise ValueError(f"All values must be greater than zero. Got {v}")
        return v

    @pydantic.field_validator("p", mode="before")
    @classmethod
    def _normalize_p(cls, p: list[float]) -> list[float]:
        norm = sum(p)
        return [x / norm for x in p]

    def crop(self):
        """Reduce the envelope length by one."""
        self.M = [x for x in self.M[:-1]]
        norm = sum(self.p[:-1])
        self.p = [x / norm for x in self.p[:-1]]

    def get_mmi_mass(self):
        """Retrieve the envelope MMI."""
        return self.M[0]

    def _get_mass_tolerance(self, k: int, tol: EnvelopeQueryTolerance) -> float:
        slope = tol.max_M_tol - tol.min_M_tol
        intercept = tol.min_M_tol
        return (1 - self.p[k]) * slope + intercept

    def get_bounds(self, tol: EnvelopeQueryTolerance, k: int) -> EnvelopeBounds:
        """Compute the bounds for the k-th element of the query."""
        if k > len(self.M):
            raise ValueError("`k` must be lower than the length of the query envelope.")
        m_tol = self._get_mass_tolerance(k, tol)
        Mk = self.M[k]
        pk = self.p[k]
        m_bounds = Bounds(lower=min(Mk - m_tol, 0.0), upper=Mk + m_tol)
        p_bounds = Bounds(lower=min(pk - tol.p_tol, 0.0), upper=pk + tol.p_tol)
        return EnvelopeBounds(M=m_bounds, p=p_bounds, index=k)


class Envelope(pydantic.BaseModel):
    """Store the envelope of molecular species."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    M: FloatArray
    """The exact mass values of the species."""

    p: FloatArray
    """The normalized abundance of the species."""

    include: IntArray1D | None = None
    """A list of row indices to include in the envelope. If set to ``None`` include all rows."""

    def iterate_rows(self) -> Generator[tuple[FloatArray1D, FloatArray1D], None, None]:
        """Iterate over rows of both arrays."""
        indices = np.arange(self.get_n_rows()) if self.include is None else self.include

        for i in indices:
            yield self.M[i], self.p[i]

    def get_n_rows(self) -> int:
        """Retrieve the current number of included rows."""
        return self.M.shape[0] if self.include is None else self.include.size

    def get_M_column(self, k):
        """Fetch the k-th column from the mass array."""
        if self.include is None:
            res = self.M[:, k]
        else:
            res = self.M[self.include, k]
        return res

    def get_p_column(self, k):
        """Fetch the k-th column from the abundance array."""
        if self.include is None:
            res = self.p[:, k]
        else:
            res = self.p[self.include, k]
        return res

    def exclude(self, bounds: EnvelopeBounds) -> None:
        """Remove envelopes outside the provided bounds from included envelopes."""
        include = np.arange(self.get_n_rows()) if self.include is None else self.include.copy()

        Mk = self.get_M_column(bounds.index)
        valid_M = (Mk >= bounds.M.lower) & (Mk <= bounds.M.upper)

        pk = self.get_p_column(bounds.index)
        valid_p = (pk >= bounds.p.lower) & (pk <= bounds.p.upper)
        include = include[valid_p & valid_M]

        self.include = include

    def reset_exclude(self):
        """Include all excluded envelopes."""
        self.include = None

    def crop(self, size: int) -> None:
        """Remove columns from mass and abundance arras starting at `size` and renormalize abundances."""
        if size < 2:
            msg = f"Minimum size is 2. Got {size}"
            raise ValueError(msg)

        if size >= self.M.shape[0]:
            return

        self.M = self.M[:, :size]
        self.p = self.p[:, :size]
        normalization = np.sum(self.p, axis=1)
        normalization = normalization.reshape((normalization.size, 1))
        self.p = self.p / normalization


class HomoAtomicEnvelopeCache:
    """Cache the isotopic envelope of homoatomic species.

    :param table: the periodic table used to fetch element data
    :param length: the length of the envelopes.

    """

    def __init__(self):
        pass

    def get_empty(self, length: int = 10) -> Envelope:
        """Get an empty envelope."""
        return _get_n_isotopes_envelope(0.0, 0, length)

    @cache
    def get_envelope(self, x: Element | Isotope, n: int, length: int = 10) -> Envelope:
        """Retrieve the isotopic envelope from a homoatomic species."""
        if n == 0:
            return _get_n_isotopes_envelope(0.0, 0, length)

        if isinstance(x, Isotope):
            return _get_n_isotopes_envelope(x.m, n, length)

        return _get_n_atoms_envelope(x, n, length)

    def make_envelope_array(self, x: Isotope | Element, n_min: int, n_max: int, length: int = 10) -> Envelope:
        """Create an array of exact mass and abundance for homonuclear formulas.

        :param isotope: Isotope
        :param table: PeriodicTable
        :param n_min: minimum formula coefficient
        :param n_max: maximum formula coefficient
        :max_length: max length of the envelope
        :return: a tuple containing a 2D array of envelope exact masses for each n value and a 2D array of abundances.

        """
        rows = n_max - n_min + 1
        M_arr = np.zeros((rows, length))
        p_arr = np.zeros((rows, length))
        for k in range(n_min, n_max + 1):
            k_env = self.get_envelope(x, k, length)
            M_arr[k - n_min] = k_env.M
            p_arr[k - n_min] = k_env.p
        return Envelope(M=M_arr, p=p_arr)


def combine_envelopes(env1: Envelope, env2: Envelope) -> Envelope:
    """Combine the exact mass and abundance of two envelopes.

    All arrays must be 2-dimensional and have the same shape.

    """
    shape = env1.M.shape
    M = np.zeros(shape, dtype=float)
    p = np.zeros(shape, dtype=float)
    # Ignore zero division errors when normalizing by pk
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(shape[1]):
            pk = (env1.p[:, : k + 1] * env2.p[:, k::-1]).sum(axis=1)
            k1 = k + 1
            k2 = k
            Mk = (env1.p[:, :k1] * env1.M[:, :k1] * env2.p[:, k2::-1]) + (
                env1.p[:, :k1] * env2.M[:, k2::-1] * env2.p[:, k2::-1]
            )
            M[:, k] = Mk.sum(axis=1) / pk
            p[:, k] = pk
    np.nan_to_num(M, copy=False)
    return Envelope(M=M, p=p)


def _get_n_atoms_envelope(element: Element, n: int, max_length: int) -> Envelope:
    """Compute the envelope of n atoms.

    aux function to _get_n_atoms_envelope.

    """
    n_isotopes = len(element.isotopes)
    # find combinations of isotopes that sum n
    combinations = _find_n_isotope_combination(n_isotopes, n)

    # find m, M and p for each combination of isotopes
    multinomial_dist = multinomial(n, [x.p for x in element.isotopes])
    m = np.matmul(combinations, [x.a for x in element.isotopes])
    M = np.matmul(combinations, [x.m for x in element.isotopes])
    p = multinomial_dist.pmf(combinations)

    # sort by exact mass
    sorted_index = np.argsort(M)
    m, M, p = m[sorted_index], M[sorted_index], p[sorted_index]

    # merge values with the same nominal mass
    _, first_occurrence = np.unique(m, return_index=True)
    m_unique = np.zeros(max_length, dtype=m.dtype)
    M_unique = np.zeros(max_length, dtype=M.dtype)
    p_unique = np.zeros(max_length, dtype=p.dtype)

    # add the length of m_unique to include all nominal mass values
    n_unique = first_occurrence.size
    first_occurrence = list(first_occurrence)
    first_occurrence.append(m.size)
    m0 = m[0]
    for k in range(max_length):
        if k < n_unique:
            start = first_occurrence[k]
            end = first_occurrence[k + 1]
            mk = m[start]
            i = mk - m0
            if i < max_length:
                m_unique[i] = mk
                pk = np.sum(p[start:end])
                p_unique[i] = pk
                M_unique[i] = np.sum(M[start:end] * p[start:end]) / pk
    p_unique = p_unique / np.sum(p_unique)
    return Envelope(M=M_unique.reshape(1, M_unique.size), p=p_unique.reshape(1, p_unique.size))


def _find_n_isotope_combination(n_isotopes, n):
    """Find combinations of isotopes such that the sum is n.

    aux function to _find_n_atoms_abundances.

    """
    n_ranges = [range(x) for x in ([n + 1] * n_isotopes)]
    combinations = cartesian_product_from_iterable(*n_ranges)
    assert combinations is not None
    valid_combinations = combinations.astype(int).sum(axis=1) == n
    combinations = combinations[valid_combinations, :]
    return combinations


def _get_n_isotopes_envelope(M: float, n: int, length: int) -> Envelope:
    """Create the isotopic envelope for n isotopes.

    :param M: the exact mass of the isotope
    :param n: the number of isotopes in the species
    :param length: the envelope length

    aux function to _get_n_atoms_envelope.

    """
    M_arr = np.zeros((1, length), dtype=float)
    p_arr = np.zeros((1, length), dtype=float)
    M_arr[0, 0] = M * n
    p_arr[0, 0] = 1.0
    return Envelope(M=M_arr, p=p_arr)
