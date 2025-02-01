"""Utilities to simulate LC-MS data.

Provides:

SimulatedLCMSDataReader
    A Reader that can be plugged into MSData to generate simulated LC-MS data files.
SimulatedLCMSSampleFactory
    A pydantic model that creates simulated sample data configuration.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pydantic
from typing_extensions import Self

from ..chem import EM, Formula
from ..core.models import Chromatogram, MSSpectrum, Sample, SampleMetadata
from ..io.reader import reader_registry
from ..utils.numpy import FloatArray1D


@reader_registry.register()
class SimulatedLCMSDataReader:
    """Read simulated LC-MS data files."""

    def __init__(self, src: pathlib.Path | Sample) -> None:
        if isinstance(src, pathlib.Path) or src.meta is None:
            msg = "Simulated LC-MS sample only work with sample models created with the simulated sample factory."
            raise ValueError(msg)
        config: SimulatedLCMSDataConfiguration = getattr(src.meta, "config")
        features: list[SimulatedLCMSFeature] = getattr(src.meta, "features")
        self._sample = SimulatedLCMSSample(config=config, features=features)
        self.spectrum_factory = MSSpectrumFactory(self._sample)

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram from file."""
        raise NotImplementedError

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from file."""
        return self.spectrum_factory.create(index)

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatograms."""
        raise NotImplementedError

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra."""
        return self._sample.config.n_scans


class SimulatedLCMSDataConfiguration(pydantic.BaseModel):
    """Store configuration of a simulated LC-MS sample."""

    grid: MZGridSpecification | None = None
    """The m/z grid specification. If not specified, a grid is created using features m/z."""

    mz_noise: pydantic.NonNegativeFloat = 0.001
    """Additive noise added to m/z in each scan"""

    amp_noise: pydantic.NonNegativeFloat = 10.0
    """additive noise added to spectral intensity on each scan"""

    mz_width: pydantic.PositiveFloat = 0.005
    """The peak width in the m/z domain"""

    n_scans: pydantic.PositiveInt = 500
    """The number of scans in the sample"""

    time_resolution: pydantic.PositiveFloat = 1.0
    """The time spacing between scans"""

    min_signal_intensity: pydantic.PositiveFloat | None = None
    """If specified, elements in a spectrum with values lower than this parameter are removed"""

    signal_noise: pydantic.PositiveFloat = 10.0
    """Additive noise added to the spectral intensity in each scan"""

    ms_level: pydantic.PositiveInt = 1
    """The spectra MS level"""


class MZGridSpecification(pydantic.BaseModel):
    """Store specification to create an m/z grid."""

    low: pydantic.PositiveFloat = 100.0
    """The minimum m/z value in the grid"""

    high: pydantic.PositiveFloat = 1200.0
    """The maximum m/z value in the grid"""

    size: pydantic.PositiveInt = 10000
    """The number of elements in the grid"""

    def create(self) -> FloatArray1D:
        """Create a m/z grid."""
        return np.linspace(self.low, self.high, self.size)


class SimulatedLCMSSampleFactory(pydantic.BaseModel):
    """Utility that creates simulated data samples."""

    config: SimulatedLCMSDataConfiguration = SimulatedLCMSDataConfiguration()
    """The sample configuration used to simulate data."""

    adducts: list[SimulatedLCMSAdductSpec] = list()
    """the list of adducts to include in the simulated sample."""

    def __call__(self, id: str, **kwargs) -> Sample:
        """Create a new simulated sample model.

        :param id: the id for the sample
        :param kwargs: extra sample information passed to the :py:class:`tidyms2.Sample` constructor.

        """
        if "path" not in kwargs:
            kwargs["path"] = pathlib.Path(".")

        reader = SimulatedLCMSDataReader.__name__
        features: list[SimulatedLCMSFeature] = list()
        for adduct in self.adducts:
            features.extend(adduct.create_features())
        features = sorted(features, key=lambda x: x.mz)
        extra = SimulatedLCMSSample(config=self.config, features=features).model_dump()
        meta = SampleMetadata(**extra)
        return Sample(id=id, reader=reader, meta=meta, **kwargs)


class SimulatedLCMSSample(pydantic.BaseModel):
    """Create simulated LC-MS data files."""

    config: SimulatedLCMSDataConfiguration
    """The sample configuration used to simulate data."""

    features: list[SimulatedLCMSFeature] = list()
    """the list of features in the sample."""

    def make_grid(self) -> FloatArray1D:
        """Create a grid from features m/z values."""
        if self.config.grid is None:
            grid = np.array(sorted([x.mz for x in self.features]))
        else:
            grid = self.config.grid.create()
        return grid

    @classmethod
    def from_json(cls, path: pathlib.Path) -> Self:
        """Create a new instance from a JSON file."""
        with path.open("rt") as f:
            model_json = f.read()
        return cls.model_validate_json(model_json)

    def to_json(self, path: pathlib.Path):
        """Store the model as a JSON file."""
        with path.open("wt") as f:
            f.write(self.model_dump_json())


class SimulatedLCMSAdductSpec(pydantic.BaseModel):
    """Define a set of isotopologue feature created from an adduct."""

    formula: str
    """The adduct formula. Used as a :py:class:`tidyms.chem.Formula` argument."""

    n_isotopologues: pydantic.PositiveInt = 5
    """The number of isotopologues to simulate."""

    rt_mean: pydantic.PositiveFloat
    """The adduct retention time"""

    rt_noise: pydantic.PositiveFloat | None = None
    """Additive noise for the features retention time."""

    rt_width: pydantic.PositiveFloat = 3.0
    """The features peak width"""

    base_intensity: pydantic.PositiveFloat = 100.0
    """The base adduct intensity. This value is scaled by the relative abundance of isotopologues to
    compute the feature height.
    """

    def create_features(self) -> list[SimulatedLCMSFeature]:
        """Create a list of simulated features to simulate a sample."""
        formula = Formula(self.formula)
        envelope = formula.get_isotopic_envelope(n=self.n_isotopologues)

        if self.rt_noise is None:
            rt_noise = 0.0
        else:
            rt_noise = np.random.normal(scale=self.rt_noise)

        rt = self.rt_mean + rt_noise

        features = list()
        for Mk, pk in zip(envelope.mz, envelope.p):
            mzk = (Mk - formula.charge * EM) / abs(formula.charge)
            int_k = self.base_intensity * pk
            ft = SimulatedLCMSFeature(mz=mzk, rt=rt, int=int_k, width=self.rt_width)
            features.append(ft)
        return features


class SimulatedLCMSFeature(pydantic.BaseModel):
    """Store a simulated LC-MS peak information."""

    mz: pydantic.PositiveFloat
    """The feature m/z."""

    rt: pydantic.PositiveFloat
    """The feature retention time."""

    int: pydantic.PositiveFloat
    """the feature intensity."""

    width: pydantic.PositiveFloat
    """The peak width in the time domain"""


class MSSpectrumFactory:
    """Store m/z grid data."""

    def __init__(self, sample: SimulatedLCMSSample) -> None:
        self.sample = sample
        self.grid = sample.make_grid()
        self._is_centroid = sample.config.grid is not None
        scan = sample.config.n_scans
        self._random_seeds = np.random.choice(scan * 10, scan)

    def create(self, scan: int) -> MSSpectrum:
        """Create a ms spectrum instance."""
        assert scan < self.sample.config.n_scans, "`scan` must be lower than the sample `n_scans` parameter."

        mz = self._compute_mz(scan)
        sp = self._compute_intensity(mz, scan)

        if self.sample.config.min_signal_intensity is not None:
            mask = sp >= self.sample.config.min_signal_intensity
            mz = mz[mask]
            sp = sp[mask]

        time = self.sample.config.time_resolution * scan

        return MSSpectrum(
            index=scan, mz=mz, int=sp, ms_level=self.sample.config.ms_level, centroid=self._is_centroid, time=time
        )

    def _compute_mz(self, scan: int) -> FloatArray1D:
        # use the same random seed for a given scan for reproducibility
        seed = self._random_seeds[scan]
        np.random.seed(seed)

        # add random noise to the m/z grid
        noise_level = self.sample.config.mz_noise

        if noise_level > 0.0:
            noise = np.random.normal(size=self.grid.size, scale=noise_level)
            mz = self.grid + noise
        else:
            mz = self.grid.copy()

        return mz

    def _compute_intensity(self, mz: FloatArray1D, scan: int):
        time = self.sample.config.time_resolution * scan
        intensity = np.zeros_like(mz)
        mz_width = self.sample.config.mz_width
        for ft in self.sample.features:
            amp = ft.int * np.power(np.e, -0.5 * ((time - ft.rt) / ft.width) ** 2)
            intensity += amp * np.power(np.e, -0.5 * ((mz - ft.mz) / mz_width) ** 2)

        if self.sample.config.amp_noise > 0.0:
            intensity += np.random.normal(size=intensity.size, scale=self.sample.config.amp_noise)
            intensity[intensity < 0] = 0.0

        return intensity
