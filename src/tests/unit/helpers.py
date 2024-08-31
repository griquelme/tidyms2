"""Helpers classes and functions for unit tests."""

from __future__ import annotations

from math import isnan, nan
from pathlib import Path
from random import randint, random

from typing_extensions import Self

from tidyms2.core import models, operators
from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode
from tidyms2.core.models import AnnotableFeature, IsotopicEnvelope, Roi, Sample


class ConcreteRoi(Roi):
    data: list[float] = list()

    def equal(self, other: ConcreteRoi) -> bool:
        """Compare equal."""
        return self.data == other.data


class ConcreteFeature(AnnotableFeature[ConcreteRoi]):
    data: int
    custom_descriptor: float = nan

    def _set_custom_descriptor(self):
        self.custom_descriptor = 100.0

    def _set_area(self):
        if isnan(self.area):
            self.area = 100.0 * self.data

    def _set_mz(self):
        # make mz depend on data field to test equal method
        if isnan(self.mz):
            self.mz = 100.0 * self.data

    def _set_height(self):
        if isnan(self.height):
            self.height = 50.0 * self.data

    def equal(self, other: ConcreteFeature) -> bool:
        return self.data == other.data

    def compare(self, other: Self) -> float:
        return 1.0

    @staticmethod
    def compute_isotopic_envelope(*features) -> IsotopicEnvelope:
        total_area = sum([x.area for x in features])
        mz = [x.mz for x in features]
        p = [x.area / total_area for x in features]
        return IsotopicEnvelope(mz=mz, p=p)


class DummyRoiExtractor(operators.RoiExtractor[ConcreteRoi, ConcreteFeature]):
    n_roi: int = 3
    """The number of dummy ROI to extract."""

    param2: str = "default"

    def extract_rois(self, sample: Sample) -> list[ConcreteRoi]:
        return [create_roi(sample) for _ in range(self.n_roi)]

    def pre_apply(self):
        """Test pre apply functionality."""
        pass

    def post_apply(self):
        """Test post apply functionality."""
        pass

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


class DummyRoiTransformer(operators.RoiTransformer[ConcreteRoi, ConcreteFeature]):
    max_length: int = 2
    """Crop ROI length to max length. ROIs with length greater than this value are deleted."""

    param2: str = "default"

    def transform_roi(self, roi: ConcreteRoi) -> ConcreteRoi | None:
        if len(roi.data) <= self.max_length:
            roi.data = roi.data[: self.max_length]
            return roi
        return None

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


class DummyFeatureExtractor(operators.FeatureExtractor[ConcreteRoi, ConcreteFeature]):
    n_features: int = 2
    """The number of features to extract from each ROI."""

    param2: str = "default"

    def extract_features(self, roi: ConcreteRoi) -> list[ConcreteFeature]:
        return [create_feature(roi) for _ in range(self.n_features)]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


class DummyFeatureTransformer(operators.FeatureTransformer[ConcreteRoi, ConcreteFeature]):
    feature_value: int = 5
    """The value to set in feature data."""

    param2: str = "default"

    def transform_feature(self, feature: ConcreteFeature) -> ConcreteFeature | None:
        if feature.data <= self.feature_value:
            feature.data = self.feature_value
            return feature
        return None

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


# @processors.ProcessorRegistry.register
# class DummyFeatureTransformer(processors.FeatureTransformer):
#     param1: float = 10.0
#     param2: bool = False

#     def _transform_feature(self, feature: Feature):
#         pass

#     def get_default_parameters(
#         self,
#         instrument: c.MSInstrument = c.MSInstrument.QTOF,
#         separation: c.SeparationMode = c.SeparationMode.UPLC,
#         polarity: c.Polarity = c.Polarity.POSITIVE,
#     ):
#         return dict()


def create_sample(path: Path, suffix: int, group: str = "") -> models.Sample:
    file = path / f"sample-{suffix}.mzML"
    file.touch()
    sample = models.Sample(path=file, id=file.stem, group=group, order=suffix)
    return sample


def create_roi(sample: Sample) -> ConcreteRoi:
    data = [random() for _ in range(5)]
    return ConcreteRoi(data=data, sample=sample)


def create_feature(roi: ConcreteRoi) -> ConcreteFeature:
    data = randint(0, 10)
    return ConcreteFeature(roi=roi, data=data, height=float(data))


# def add_dummy_features(roi_list: list[ConcreteRoi], n: int):
#     label_counter = 0
#     for roi in roi_list:
#         for _ in range(n):
#             annotation = models.Annotation(group=label_counter)
#             ft = create_feature(roi, annotation)
#             roi.add_feature(ft)
#             label_counter += 1


# def get_feature_list(roi_list: list[ConcreteRoi]) -> list[ConcreteFeature]:
#     feature_list = list()
#     for roi in roi_list:
#         feature_list.extend(roi.features)
#     return feature_list


# def create_dummy_sample_data(
#     path: Path, suffix: int, group: str = "", with_roi: bool = False, with_feature: bool = False
# ) -> SampleData:
#     """
#     Create a dummy sample data instance.

#     Parameters
#     ----------
#     path : Path
#         dummy path pointing to raw data.
#     suffix : int
#         numeric suffix appended to sample id.
#     group : str, optional
#         The group where the sample belongs to., by default ""
#     with_roi : bool, optional
#         If ``True``, adds dummy ROIs to sample, by default False
#     with_feature : bool, optional
#         If ``True`` add dummy features to ROI, by default False

#     Returns
#     -------
#     SampleData

#     """
#     sample = create_dummy_sample(path, suffix, group)

#     if with_roi:
#         roi_list = [create_dummy_roi() for _ in range(5)]
#     else:
#         roi_list = list()

#     if with_roi and with_feature:
#         add_dummy_features(roi_list, 5)

#     return SampleData(sample=sample, roi=roi_list)
