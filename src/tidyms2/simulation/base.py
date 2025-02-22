"""Base utilities for simulation."""

from __future__ import annotations

import random
from functools import cache, cached_property
from math import exp

import numpy
import pydantic

from ..chem import EM, Formula
from ..core.models import IsotopicEnvelope
from ..core.utils.numpy import FloatArray1D


class DataAcquisitionSpec(pydantic.BaseModel):
    """Define the acquisition parameters of a simulated."""

    grid: MZGridSpec | None = None
    """The m/z grid specification. If not specified, a grid is created using features m/z"""

    mz_std: pydantic.NonNegativeFloat = 0.0
    """Additive noise added to m/z in each scan"""

    int_std: pydantic.NonNegativeFloat = 0.0
    """additive noise added to spectral intensity on each scan"""

    mz_width: pydantic.PositiveFloat = 0.005
    """The peak width in the m/z domain. Used only when a grid specification is provided"""

    n_scans: pydantic.PositiveInt = 100
    """The number of scans in the sample"""

    time_resolution: pydantic.PositiveFloat = 1.0
    """The time spacing between scans"""

    min_int: pydantic.PositiveFloat | None = None
    """If specified, elements in a spectrum with intensity values lower than this parameter are removed"""

    ms_level: pydantic.PositiveInt = 1
    """The spectra MS level"""


class MZGridSpec(pydantic.BaseModel):
    """Define the minimum, maximum and spacing of m/z values in spectra."""

    low: pydantic.PositiveFloat = 100.0
    """The minimum m/z value in the grid"""

    high: pydantic.PositiveFloat = 1200.0
    """The maximum m/z value in the grid"""

    size: pydantic.PositiveInt = 10000
    """The number of elements in the grid"""

    def create(self) -> FloatArray1D:
        """Create a m/z grid."""
        return numpy.linspace(self.low, self.high, self.size)


class AbundanceSpec(pydantic.BaseModel):
    """Define the abundance of a chemical species in a sample.

    The abundance is computed as follows:

    1.  draw a value :math:`u` from the uniform distribution :math:`[0, 1]`.
    2.  If :math:`u` is lower than the `prevalence` field, set the abundance to zero.
    3.  Otherwise, set the abundance to a value sampled from a gaussian distribution.

    """

    mean: pydantic.PositiveFloat = 100.0
    """The mean of the gaussian distribution used to sample the abundance."""

    std: pydantic.PositiveFloat = 0.0
    """The standard deviation of the gaussian distribution used to sample the abundance"""

    prevalence: float = pydantic.Field(gt=0.0, le=1.0, default=1.0)
    """The probability of the species to be present in a sample."""

    def sample_abundance(self) -> float:
        """Get a realization of the abundance."""
        is_in_sample = random.uniform(0.0, 1.0) < self.prevalence
        c = max(0.0, random.gauss(mu=self.mean, sigma=self.std))  # force signal to be non-negative
        return c if is_in_sample else 0.0


class MeasurementNoiseSpec(pydantic.BaseModel):
    r"""Define an additive error term added to the measured signal of an adduct.

    The noise for a sample is computed as follows:

    1.  compute the :term:`snr` using the `base_snr` field and the isotopologue abundance as a
        scaling factor. Set the :term:`snr` to the maximum value between this value and the `min_snr`
        field.
    2.  Compute the noise level :math:`\sigma:` as the quotient between the signal and the :term:`snr`.
    3.  Sample the noise from a distribution :math:`~N(0, \sigma)`

    """

    base_snr: pydantic.PositiveFloat | None = None
    """The base :term:`snr` of the additive noise applied to isotopologue signal"""

    min_snr: pydantic.PositiveFloat = 10.0
    """The minimum :term:`snr` of the additive noise applied to isotopologues signal. This value allows
    to set a lower bound on the :term:`snr` for low intensity features. If the `snr` parameter is
    not set, this value is ignored."""

    @pydantic.model_validator(mode="after")
    def _ensure_snr_geq_min_snr(self):
        if self.base_snr is not None and self.base_snr < self.min_snr:
            raise ValueError("`snr` field must be greater or equal than `min_snr`.")
        return self

    def compute_snr(self, pk: float) -> float | None:
        """Compute the :term:`snr` level for an isotopologue signal.

        :param pk: the isotopologue relative abundance, used to scale the base :term:`snr`

        """
        return None if self.base_snr is None else max(self.min_snr, self.base_snr * pk)

    def sample_noise(self, signal: float, pk: float) -> float:
        """Draw a noise term sample for an isotopologue signal.

        :param signal: the observed signal, as the product of the instrument
            response, the base abundance and the isotopologue relative abundance.
        :param pk: the relative abundance of an isotopologue, used to scale the :term:`snr`
            for isotopologues with lower signals.

        """
        snr = self.compute_snr(pk)
        return 0.0 if snr is None else random.gauss(sigma=signal / snr)


class InstrumentResponseSpec(pydantic.BaseModel):
    r"""Define the instrument response factor for an adduct.

    Computed as the product of the base response factor, the inter-batch effect factor
    and the sensitivity loss factor.

    The default parameters of this specification will generate a response factor without
    sensitivity loss over time and no additive noise.

    Refer to the :ref:`simulation-guide` for more details.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    base_response_factor: pydantic.PositiveFloat = 1.0
    """The adduct base response when no inter-batch or sensitivity loss effects are present."""

    max_sensitivity_loss: float = pydantic.Field(ge=0.0, le=1.0, default=0.0)
    """The maximum sensitive loss in an analytical batch."""

    sensitivity_decay: float = pydantic.Field(ge=0.0, default=0.0)
    """The decay parameter for the time-dependent sensitivity loss. We suggest to use values between
    ``0.001`` and ``1.0``, as larger values decays to the maximum sensitivity value too fast. This
    value should also be selected based on the batch size, as in longer batches it is possible
    to see the effect of smaller decay values.
    """

    interbatch_variation: float = pydantic.Field(ge=0.0, le=1.0, default=1.0)
    """A factor applied to all samples from the same analytical batch. The interbatch
    variation factor is random but equal for all observations within a batch. The value
    for a given batch is samples from a uniform distribution with minimum value equal to
    this parameter and maximum value equal to ``1.0``. In the default configuration the
    inter-batch factor is set to ``1.0`` always."""

    @cache
    def get_interbatch_factor(self, batch: int) -> float:
        return random.uniform(self.interbatch_variation, 1.0)

    def get_sensitivity_loss_factor(self, order: int) -> float:
        """Compute the sensitivity loss factor applied to the base response."""
        sensitivity_decay = self.max_sensitivity_loss * exp(-order * self.sensitivity_decay)
        return 1 - self.max_sensitivity_loss + sensitivity_decay

    def compute_response_factor(self, order: int, batch: int) -> float:
        """Compute the response factor for a specific sample run order and analytical batch.

        :param order: the relative run order within a batch.
        :param batch: the analytical batch number

        """
        interbatch_factor = self.get_interbatch_factor(batch)
        sensitivity_loss = self.get_sensitivity_loss_factor(order)
        return self.base_response_factor * sensitivity_loss * interbatch_factor


class BaseChemicalSpeciesSpec(pydantic.BaseModel):
    r"""Define the how the signal generated by a chemical species are computed.

    .. math::

        x_{j} = c * p_{j} * f + \epsilon

    Where :math:`x_{j}` is the signal intensity for the j-th isotopologue, :math:`c` is the
    abundance of the species that generates the adducts, :math:`p_{j}` is the abundance of the
    j-th isotopologue included in the simulation, :math:`f` is the response factor of the instrument
    and :math:`epsilon` is an additive error term.

    """

    formula: str
    """The ion formula. Used as a :py:class:`tidyms.chem.Formula` argument."""

    n_isotopologues: pydantic.PositiveInt = 1
    """The number of isotopologues to simulate."""

    abundance: dict[str, AbundanceSpec] | AbundanceSpec = AbundanceSpec()
    """Define the abundance :math:`c` of the chemical species that generates the ion. Multiple
    abundance specifications may be defined for different sample groups. In this case, the
    corresponding specification will be selected based on the simulated sample group. If the
    sample groups is not found a ``ValueError`` will be raised."""

    response: InstrumentResponseSpec = InstrumentResponseSpec()
    """Define how the response factor :math:`f` is computed"""

    noise: MeasurementNoiseSpec = MeasurementNoiseSpec()
    """Defines how the additive noise :math:`epsilon` is computed"""

    @pydantic.computed_field(repr=False)
    @cached_property
    def charge(self) -> int:
        return Formula(self.formula).charge

    @pydantic.computed_field(repr=False)
    @cached_property
    def _envelope(self) -> IsotopicEnvelope:
        return Formula(self.formula).get_isotopic_envelope(self.n_isotopologues)

    def compute_abundance(self, group: str | None = None) -> float:
        """Compute a realization of the species abundance in the specified group.

        :param group: the group name if multiple groups where provided for the abundance specification.
            If not provided a default group is chosen.
        :raises ValueError: if the group is not found in the specification

        """
        if group is None and isinstance(self.abundance, dict):
            try:
                group = list(self.abundance)[0]
            except IndexError as e:
                raise ValueError("Abundance spec cannot be an empty dictionary") from e

        if isinstance(self.abundance, dict):
            assert isinstance(group, str)
            try:
                return self.abundance[group].sample_abundance()
            except KeyError as e:
                raise ValueError(f"Group `{group}` is not defined in the abundance specification.") from e

        return self.abundance.sample_abundance()

    def get_mz(self) -> list[float]:
        """Compute the m/z of features in the adduct."""
        return [(Mk - self.charge * EM) / abs(self.charge) for Mk in self._envelope.mz]

    def compute_intensity(self, group: str | None = None, order: int = 0, batch: int = 0) -> list[float]:
        """Compute a realization of features intensity all isotopologues."""
        c = self.compute_abundance(group)
        intensity = list()
        for pk in self._envelope.p:
            f = self.response.compute_response_factor(order, batch)
            signal = c * f * pk
            noise = self.noise.sample_noise(signal, pk)
            x = max(0.0, signal + noise)  # force intensity to be non-negative
            intensity.append(x)
        return intensity
