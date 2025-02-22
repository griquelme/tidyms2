"""Utilities to simulate LC-MS data."""

from __future__ import annotations

import pathlib
import random

import numpy
import pydantic

from ..core.matrix import DataMatrix
from ..core.models import Chromatogram, FeatureGroup, GroupAnnotation, MSSpectrum, Sample, SampleMetadata
from ..core.utils.numpy import FloatArray, FloatArray1D
from ..io.reader import reader_registry
from .base import BaseChemicalSpeciesSpec, DataAcquisitionSpec


@reader_registry.register()
class SimulatedLCMSDataReader:
    """Read simulated LC-MS data files."""

    def __init__(self, src: pathlib.Path | Sample) -> None:
        if isinstance(src, pathlib.Path) or not hasattr(src.meta, "_simulated"):
            msg = "Simulated LC-MS sample only work with sample models created with the simulated sample factory."
            raise ValueError(msg)
        self._simulated_sample_spec = src.meta._simulated  # type: ignore
        self._spectrum_factory = MSSpectrumFactory(self._simulated_sample_spec)

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram from file."""
        raise NotImplementedError

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from file."""
        return self._spectrum_factory.create(index)

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatograms."""
        raise NotImplementedError

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra."""
        return self._simulated_sample_spec.config.n_scans


class SimulatedLCMSSampleFactory(pydantic.BaseModel):
    """Utility that creates simulated data samples.

    Refer to the :ref:`simulation-guide` guide for a tutorial on how to use this class.

    """

    data_acquisition: DataAcquisitionSpec = DataAcquisitionSpec()
    """The sample configuration used to simulate data."""

    adducts: list[SimulatedLCMSAdductSpec] = list()
    """the list of adducts to include in the simulated sample."""

    def __call__(self, id: str, group: str | None = None, order: int = 0, **kwargs) -> Sample:
        """Create a :py:class:`~tidyms2.core.models.Sample` with simulated data specs.

        Refer to the :ref:`simulation-guide` guide for a tutorial on how to use this function.

        :param id: the id for the sample
        :param kwargs: extra sample information passed to the :py:class:`tidyms2.Sample` constructor.

        """
        if "path" not in kwargs:
            kwargs["path"] = pathlib.Path(".")

        reader = SimulatedLCMSDataReader.__name__
        meta = SampleMetadata(group=group or "", order=order)
        meta._simulated = self.create_simulated_sample_spec(group, order)  # type: ignore
        return Sample(id=id, reader=reader, meta=meta, **kwargs)

    def create_features(self, group: str | None = None, order: int = 0) -> list[SimulatedLCMSFeature]:
        """Create the list of features in a sample using the spec."""
        features: list[SimulatedLCMSFeature] = list()
        for adduct in self.adducts:
            features.extend(adduct.create_simulated_features(group, order))
        return sorted(features, key=lambda x: x.mz)

    def create_simulated_sample_spec(self, group: str | None = None, order: int = 0) -> SimulatedLCMSSample:
        """Create a simulated sample specification."""
        features = self.create_features(group, order)
        return SimulatedLCMSSample(config=self.data_acquisition, features=features)


class SimulatedLCMSSample(pydantic.BaseModel):
    """Create simulated LC-MS data files."""

    config: DataAcquisitionSpec
    """The sample configuration used to simulate data."""

    features: list[SimulatedLCMSFeature] = list()
    """the list of features in the sample."""

    def make_grid(self) -> FloatArray1D:
        """Create a grid from features m/z values."""
        if self.config.grid is None:
            grid = numpy.array(sorted([x.mz for x in self.features]))
        else:
            grid = self.config.grid.create()
        return grid


class RtSpec(pydantic.BaseModel):
    """Define the retention time value in samples."""

    mean: pydantic.PositiveFloat = 10.0
    """The mean retention time across samples"""

    std: pydantic.NonNegativeFloat = 0.0
    """The retention time standard deviation across samples"""

    width: pydantic.PositiveFloat = 3.0
    """The peak width of the chromatogram"""

    def compute_sample_rt(self) -> float:
        """Compute an observation of the retention time."""
        noise = random.gauss(sigma=self.std)
        return max(0.0, self.mean + noise)


class SimulatedLCMSAdductSpec(BaseChemicalSpeciesSpec):
    """Define a set of isotopologue feature created from an adduct."""

    rt: RtSpec = RtSpec()
    """The adduct retention time specification"""

    def create_simulated_features(self, group: str | None = None, order: int = 0) -> list[SimulatedLCMSFeature]:
        """Create a list of simulated features used by a simulated sample."""
        rt = self.rt.compute_sample_rt()
        features = list()
        for mzk, ik in zip(self.get_mz(), self.compute_intensity(group, order)):
            ft = SimulatedLCMSFeature(mz=mzk, rt=rt, int=ik, width=self.rt.width)
            features.append(ft)
        return features


class SimulatedLCMSFeature(pydantic.BaseModel):
    """Store a simulated LC-MS peak information."""

    mz: pydantic.PositiveFloat
    """The feature m/z."""

    rt: pydantic.PositiveFloat
    """The feature retention time."""

    int: pydantic.NonNegativeFloat
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
        self._random_seeds = numpy.random.choice(scan * 10, scan)

    def create(self, scan: int) -> MSSpectrum:
        """Create a ms spectrum instance."""
        assert scan < self.sample.config.n_scans, "`scan` must be lower than the sample `n_scans` parameter."

        mz = self._compute_mz(scan)
        sp = self._compute_intensity(mz, scan)

        if self.sample.config.min_int is not None:
            mask = sp >= self.sample.config.min_int
            mz = mz[mask]
            sp = sp[mask]

        time = self.sample.config.time_resolution * scan

        return MSSpectrum(
            index=scan, mz=mz, int=sp, ms_level=self.sample.config.ms_level, centroid=self._is_centroid, time=time
        )

    def _compute_mz(self, scan: int) -> FloatArray1D:
        # use the same random seed for a given scan for reproducibility
        seed = self._random_seeds[scan]
        numpy.random.seed(seed)

        # add random noise to the m/z grid
        noise_level = self.sample.config.mz_std

        if noise_level > 0.0:
            noise = numpy.random.normal(size=self.grid.size, scale=noise_level)
            mz = self.grid + noise
        else:
            mz = self.grid.copy()

        return mz

    def _compute_intensity(self, mz: FloatArray1D, scan: int):
        time = self.sample.config.time_resolution * scan
        intensity = numpy.zeros_like(mz)
        mz_width = self.sample.config.mz_width
        for ft in self.sample.features:
            amp = ft.int * numpy.power(numpy.e, -0.5 * ((time - ft.rt) / ft.width) ** 2)

            # this step allows computing values both in profile and centroid mode
            intensity += amp * numpy.power(numpy.e, -0.5 * ((mz - ft.mz) / mz_width) ** 2)

        if self.sample.config.int_std > 0.0:
            intensity += numpy.random.normal(size=intensity.size, scale=self.sample.config.int_std)
            intensity[intensity < 0] = 0.0

        return intensity


def simulate_data_matrix(specs: list[SimulatedLCMSAdductSpec], samples: list[Sample]) -> DataMatrix:
    """Create a data matrix using LC-MS simulation specs.

    Refer to the :ref:`matrix-simulation-guide` guide for a tutorial on how to use this function.

    :param specs: a list of adduct specifications to simulate features.
    :param samples: the list of samples to include in the matrix.
    :return: the simulated data matrix

    """
    if not samples:
        raise ValueError("At least one sample is required to simulate a data matrix.")

    # sorting samples by order is required to compute intra-batch effects
    samples = sorted(samples, key=lambda x: x.meta.order)

    X = _compute_matrix_values(specs, samples)
    feature_groups = _compute_feature_groups(specs)

    return DataMatrix(samples, feature_groups, X)


def _compute_matrix_values(specs: list[SimulatedLCMSAdductSpec], samples: list[Sample]) -> FloatArray:
    n_features = sum(x.n_isotopologues for x in specs)
    n_samples = len(samples)
    X = numpy.zeros(shape=(n_samples, n_features), dtype=float)

    first_sample = samples[0]
    previous_sample_batch = first_sample.meta.batch
    batch_start_at = first_sample.meta.order

    for k, sample in enumerate(samples):
        row = list()

        if previous_sample_batch != sample.meta.batch:
            batch_start_at = sample.meta.order
            previous_sample_batch = sample.meta.batch
        intra_batch_order = sample.meta.order - batch_start_at

        for spec in specs:
            row.extend(
                spec.compute_intensity(
                    group=sample.meta.group,
                    order=intra_batch_order,
                    batch=sample.meta.batch,
                )
            )
        X[k] = row
    return X


def _compute_feature_groups(specs: list[SimulatedLCMSAdductSpec]) -> list[FeatureGroup]:
    feature_groups = list()
    group_counter = 0

    for isotopologue_group, spec in enumerate(specs):
        for isotopologue_index, ft in enumerate(spec.create_simulated_features()):
            ann = GroupAnnotation(
                label=group_counter,
                isotopologue_group=isotopologue_group,
                isotopologue_index=isotopologue_index,
                charge=spec.charge,
            )

            descriptors = {"mz": ft.mz, "rt": ft.rt}
            group = FeatureGroup(group=group_counter, annotation=ann, descriptors=descriptors)
            feature_groups.append(group)
            group_counter += 1
    return feature_groups
