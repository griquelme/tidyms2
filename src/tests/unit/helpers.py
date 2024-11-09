"""Helpers classes and functions for unit tests."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from random import randint, random

from pydantic import Field, computed_field
from typing_extensions import Self

from tidyms2.core import models, operators
from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode
from tidyms2.core.models import AnnotableFeature, IsotopicEnvelope, Roi, Sample
from tidyms2.core.registry import operator_registry


class ConcreteRoi(Roi):
    data: list[float] = list()

    def equal(self, other: ConcreteRoi) -> bool:
        """Compare equal."""
        return self.data == other.data


class ConcreteFeature(AnnotableFeature[ConcreteRoi]):
    data_mz: float = Field(repr=False)
    """A proxy variable for the feature m/z."""

    data_area: float = Field(repr=False, default=100.0)
    """A proxy variable for the feature area"""

    @computed_field(repr=False)
    @cached_property
    def custom_descriptor(self) -> float:
        return 100.0

    @computed_field(repr=False)
    @cached_property
    def area(self) -> float:
        return self.data_area

    @computed_field(repr=False)
    @cached_property
    def mz(self) -> float:
        return self.data_mz

    @computed_field(repr=False)
    @cached_property
    def height(self) -> float:
        return self.data_area

    def equal(self, other: ConcreteFeature) -> bool:
        return self.data_mz == other.data_mz

    def compare(self, other: Self) -> float:
        return 1.0

    @staticmethod
    def compute_isotopic_envelope(*features) -> IsotopicEnvelope:
        total_area = sum([x.area for x in features])
        mz = [x.mz for x in features]
        p = [x.area / total_area for x in features]
        return IsotopicEnvelope(mz=mz, p=p)


@operator_registry.register()
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


@operator_registry.register()
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


@operator_registry.register()
class DummyFeatureExtractor(operators.FeatureExtractor[ConcreteRoi, ConcreteFeature]):
    n_features: int = 2
    """The number of features to extract from each ROI."""

    param2: str = "default"

    def extract_features(self, roi: ConcreteRoi) -> list[ConcreteFeature]:
        return [create_feature(roi) for _ in range(self.n_features)]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


@operator_registry.register()
class DummyFeatureTransformer(operators.FeatureTransformer[ConcreteRoi, ConcreteFeature]):
    feature_value: int = 5
    """The value to set in feature data."""

    param2: str = "default"

    def transform_feature(self, feature: ConcreteFeature) -> ConcreteFeature | None:
        if feature.data_mz <= self.feature_value:
            feature.data_mz = self.feature_value
            return feature
        return None

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()


@operator_registry.register()
class DummyAnnotationPatcher(operators.AnnotationPatcher):
    """Set annotation group to a fixed values in all features."""

    group: int = 0

    def compute_patches(self, data: operators.AssayStorage) -> list[models.AnnotationPatch]:
        annotations = data.fetch_annotations()
        return [models.AnnotationPatch(id=x.id, field="group", value=0) for x in annotations]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    def pre_apply(self):
        """Test pre apply functionality."""
        pass

    def post_apply(self):
        """Test post apply functionality."""
        pass


@operator_registry.register()
class DummyDescriptorPatcher(operators.DescriptorPatcher):
    """Set descriptor to a fixed value in all features."""

    patch: float = 500.0
    descriptor: str = "custom_descriptor"

    def compute_patches(self, data: operators.AssayStorage) -> list[models.DescriptorPatch]:
        annotations = data.fetch_annotations()
        return [models.DescriptorPatch(id=x.id, descriptor=self.descriptor, value=self.patch) for x in annotations]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    def pre_apply(self):
        """Test pre apply functionality."""
        pass

    def post_apply(self):
        """Test post apply functionality."""
        pass


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
    return ConcreteFeature(roi=roi, data_mz=data)


def create_features_from_formula(formula_str: str, sample: Sample) -> list[ConcreteFeature]:
    formula = Formula(formula_str)
    env = formula.get_isotopic_envelope()
    q = abs(formula.charge) if formula.charge else 1
    features = list()
    for Mk, pk in zip(env.mz, env.p):
        features.append(ConcreteFeature(data_mz=Mk / q, data_area=pk, roi=ConcreteRoi(sample=sample)))
    return features
