"""Base utilities for simulation."""

from __future__ import annotations

import random
from functools import cache, cached_property
from math import exp

import numpy
import pydantic

from ..chem import EM, Formula
from ..core.models import IsotopicEnvelope
from ..utils.numpy import FloatArray1D


class DataAcquisitionSpec(pydantic.BaseModel):
    """Store configuration of a simulated LC-MS sample."""

    grid: MZGridSpec | None = None
    """The m/z grid specification. If not specified, a grid is created using features m/z."""

    mz_std: pydantic.NonNegativeFloat = 0.0
    """Additive noise added to m/z in each scan"""

    int_std: pydantic.NonNegativeFloat = 0.0
    """additive noise added to spectral intensity on each scan"""

    mz_width: pydantic.PositiveFloat = 0.005
    """The peak width in the m/z domain. Used only when a grid specification is provided."""

    n_scans: pydantic.PositiveInt = 100
    """The number of scans in the sample"""

    time_resolution: pydantic.PositiveFloat = 1.0
    """The time spacing between scans"""

    min_int: pydantic.PositiveFloat | None = None
    """If specified, elements in a spectrum with values lower than this parameter are removed"""

    ms_level: pydantic.PositiveInt = 1
    """The spectra MS level"""


class MZGridSpec(pydantic.BaseModel):
    """Store specification to create an m/z grid."""

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
    """Define the abundance of a chemical species in a sample."""

    mean: pydantic.PositiveFloat = 100.0
    """The mean abundance for isotopologues in a group."""

    std: pydantic.PositiveFloat = 0.0
    """The abundance standard deviation for isotopologues in a group."""

    prevalence: float = pydantic.Field(gt=0.0, le=1.0, default=1.0)
    """The probability of the adduct occurring in a sample."""

    def compute_abundance(self) -> float:
        """Get a realization of the abundance."""
        is_in_sample = random.uniform(0.0, 1.0) < self.prevalence
        c = max(0.0, random.gauss(mu=self.mean, sigma=self.std))  # force signal to be non-negative
        return c if is_in_sample else 0.0


class MeasurementNoiseSpec(pydantic.BaseModel):
    """Define an additive error added to the measured signal."""

    snr: pydantic.PositiveFloat | None = None
    """The base :term:`snr` of the additive noise applied to isotopologues signal. This value is scaled
    the relative abundance of isotopologues. If set to ``None`` no additive noise is added."""

    max_snr: pydantic.PositiveFloat = 10.0
    """The maximum :term:`snr` of the additive noise applied to isotopologues signal. This value allows
    to set an upper bound on the :term:`snr` for low intensity features. If the `snr` parameter is
    not set, this value is ignored."""

    def compute_noise(self, signal: float, isotopologue_abundance: float) -> float:
        """Compute the noise for a signal.

        :param signal: the observed signal, computed according the response specification
        :param isotopologue_abundance: the relative abundance of an isotopologue, used to decrease
            the :term:`snr` for isotopolgoues with lower signals.

        """
        if self.snr is None:
            return 0.0
        snr = max(self.max_snr, self.snr / isotopologue_abundance)
        return random.gauss(sigma=signal / snr)


class InstrumentResponseSpec(pydantic.BaseModel):
    r"""Define how an adduct base abundance is converted into spectrum intensity.

    If :math:`c` is the abundance of the species that generated the adduct, and
    :math:`i` is the order of the sample, then the response :math:`x` is computed
    as follows:

    .. math::

        x = \tilde{f} c + \epsilon

    Where :math:`tilde{f}` is the damped response factor that accounts for instrumental
    sensitivity loss over time and is computed as follows:

    .. math::

        \tilde{f} = f b (M + (1 - M) \exp (- i \lambda))

    Where :math:`f` is the `response_factor` parameter, :math:`M` is the `max_sensitivity_loss`
    parameter and :math:`\lambda` is the sensitivity decay parameter.

    :math:`b` is the inter-batch variation, a random value sampled from a uniform distribution
    with minimum equal to the `interbatch_factor` parameter and maximum equal to 1. This value
    is sampled once for each analytical batch value and applied to all observations from that
    batch.

    The additive noise :math:`\epsilon` is sampled from a Gaussian distribution with mean 0 and
    standard deviation equal to :math:`\tilde{f} * c / \textrm{SNR}`.

    The default parameters of this specification will generate a response without sensitivity
    loss over time and no additive noise.

    """

    model_config = pydantic.ConfigDict(frozen=True)

    response_factor: pydantic.PositiveFloat = 1.0
    """The adduct instrumental response factor."""

    max_sensitivity_loss: float = pydantic.Field(ge=0.0, le=1.0, default=0.0)
    """The maximum sensitive loss in an analytical batch."""

    sensitivity_decay: float = pydantic.Field(ge=0.0, default=0.0)
    """The decay parameter for the time-dependent sensitivity loss. We suggest to use values between
    ``0.001`` and ``1.0``, as larger values decays to the maximum sensitivity value too fast. This
    value should also be selected based on the batch size, as in longer batches it is possible
    to see the effect of smaller decay values.
    """

    interbatch_variation: float = pydantic.Field(ge=0.0, le=1.0, default=1.0)
    """A factor applied to all samples from the same analytical batch."""

    @cache
    def _get_interbatch_factor(self, batch: int) -> float:
        return random.uniform(self.interbatch_variation, 1.0)

    def compute_response(self, order: int, abundance: float, batch: int) -> float:
        """Compute the measured spectral intensity.

        :param abundance: the base abundance of an isotopologue.

        """
        sensitivity_decay = (1 - self.max_sensitivity_loss) * exp(-order * self.sensitivity_decay)
        sensitivity_loss = self.max_sensitivity_loss + sensitivity_decay
        interbatch_factor = self._get_interbatch_factor(batch)
        return abundance * self.response_factor * sensitivity_loss * interbatch_factor


class BaseChemicalSpeciesSpec(pydantic.BaseModel):
    r"""Define the ionic chemical species signals in a simulated sample.

    The parameters provided in the adduct specification allow to model common
    sources of data variation in experimental data such as:

    - instrumental variations in m/z
    - sample variation in compounds abundance
    - compound prevalence
    - instrumental variation in the signal
    - time-dependent signal loss

    Each isotopologue m/z is computed using the adduct theoretical mass and charge.
    An additive noise term is added to each feature using a Gaussian distribution
    with mean set to zero and standard deviation set to `mz_std`.

    Isotopologue intensities is computed as follows. First a value is sampled from a
    uniform distribution :math:`U[0, 1]`. If this value is higher than the `prevalence`
    parameter, then all isotopologues intensities are set to zero. Otherwise, each
    isotopologue abundance is computed as follows:

    .. math::

        x = C_{0} * f + \epsilon_{I}

    Where :math:`C_{0}` for each isotopologue is computed as the product of the isotopologue
    theoretical abundance and the adduct abundance in the sample. This value is the
    realization of a Gaussian random variable with mean equal to the group abundance mean and
    standard deviation equal to the group abundance std. :math:`f` is the product of the
    `response_factor` parameter and a sensitivity loss factor computed as
    ``max_sensitivity_loss + (1 - max_sensitivity_loss) * exp(- order * sensitivity_decay)``.

    """

    formula: str
    """The ion formula. Used as a :py:class:`tidyms.chem.Formula` argument."""

    n_isotopologues: pydantic.PositiveInt = 1
    """The number of isotopologues to simulate."""

    abundance: dict[str, AbundanceSpec] | AbundanceSpec = AbundanceSpec()
    """Define the abundance distribution of the chemical species that generates the ion. Multiple
    abundance specifications may be defined for different sample groups. In this case, the
    corresponding specification will be selected based on the simulated sample group. If the
    sample groups is not found a ``ValueError`` will be raised."""

    response: InstrumentResponseSpec = InstrumentResponseSpec()
    """Defines how the isotopologues abundance is converted into the observed signal"""

    noise: MeasurementNoiseSpec = MeasurementNoiseSpec()
    """Defines the noise added to the signal"""

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

        """
        if isinstance(self.abundance, dict) and group is not None:
            return self.abundance[group].compute_abundance()
        elif isinstance(self.abundance, dict):
            msg = "A group must be specified if the abundance specifications is defined for multiple groups."
            raise ValueError(msg)

        return self.abundance.compute_abundance()

    def get_mz(self) -> list[float]:
        """Compute the m/z of features in the adduct."""
        return [(Mk - self.charge * EM) / abs(self.charge) for Mk in self._envelope.mz]

    def compute_intensity(self, group: str | None = None, order: int = 0, batch: int = 0) -> list[float]:
        """Compute a realization of features intensity for a sample."""
        c = self.compute_abundance(group)
        intensity = list()
        for pk in self._envelope.p:
            signal = self.response.compute_response(order, c * pk, batch)
            noise = self.noise.compute_noise(signal, pk)
            x = max(0.0, signal + noise)  # force intensity to be non-negative
            intensity.append(x)
        return intensity
