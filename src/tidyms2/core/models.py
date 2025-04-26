"""TidyMS core data models.

All models defined here define how data is shared in data pipelines.

Refer to the :ref:`architecture overview <overview>` for an introduction to the TidyMS data model.

For considerations on customizing these models, refer to the :ref:`extending-guide` guide.

"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Generic, Self, TypeVar
from uuid import UUID

import numpy
import pydantic

from . import exceptions
from .enums import MSDataMode
from .utils.common import create_id
from .utils.numpy import FloatArray1D, IntArray1D


class TidyMSBaseModel(pydantic.BaseModel):
    """Base model that all other library models inherit from."""

    id: UUID = pydantic.Field(default_factory=create_id)
    """A unique id for the model."""

    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class SampleMetadata(pydantic.BaseModel):
    """Sample metadata container."""

    model_config = pydantic.ConfigDict(validate_assignment=True, extra="allow")

    type: str = ""
    """the :term:`sample type`"""

    group: str = ""
    """the :term:`sample group`"""

    order: pydantic.NonNegativeInt = 0
    """the sample measurement order in an assay"""

    batch: pydantic.NonNegativeInt = pydantic.Field(default=0, repr=False)
    """the sample analytical batch number in an assay."""


class Sample(pydantic.BaseModel):
    """Store information required to load data from a raw data file."""

    model_config = pydantic.ConfigDict(validate_assignment=True, frozen=True, extra="forbid")

    path: Path
    """Path to a raw data file"""

    id: str
    """A unique sample identifier"""

    ms_level: pydantic.PositiveInt = pydantic.Field(default=1, repr=False)
    """the sample MS level"""

    reader: str | None = pydantic.Field(default=None, repr=False)
    """The name of a registered data reader to read sample data. If ``None``, the
    optimal reader is inferred from the file extension.
    """

    ms_data_mode: MSDataMode = pydantic.Field(default=MSDataMode.CENTROID, repr=False)
    """the mode in which the sample data is stored."""

    start_time: pydantic.NonNegativeFloat = pydantic.Field(default=0.0, repr=False)
    """Minimum acquisition time of MS scans to include. If ``None``, start from the first scan"""

    end_time: pydantic.NonNegativeFloat | None = pydantic.Field(default=None, repr=False)
    """Maximum acquisition time of MS scans to include. If ``None``, end at the last scan"""

    meta: SampleMetadata = pydantic.Field(default=SampleMetadata(), repr=False)
    """Sample metadata."""

    @pydantic.field_validator("path", mode="before")
    def _normalize_path(cls, value: Path) -> Path:
        if isinstance(value, str):
            value = Path(value)
        return value

    @pydantic.field_serializer("path")
    def serialize_path(self, path: Path, _info) -> str:
        """Serialize path into a string."""
        return str(path)

    @staticmethod
    def validate_samples(*samples: Sample) -> None:
        n_samples = len(samples)
        if len({x.id for x in samples}) < n_samples:
            msg = "Samples must have a unique id."
            raise exceptions.RepeatedIdError(msg)

        if len({x.meta.order for x in samples}) < n_samples:
            msg = "Samples must have a unique order."
            raise exceptions.RepeatedSampleOrder(msg)


class Roi(TidyMSBaseModel):
    """Base class for :term:`ROIs <roi>` extracted from raw MS data.

    Roi inherits from pydantic BaseModel, and support most of its functionality.
    New ROIs subclasses are created by inheritance of this class and setting
    data fields using Pydantic's standard approach.

    For Numpy array fields, check out the `tidyms.utils.numpy.FloatingArray` and
    `tidyms.utils.numpy.IntArray` types which provide type checking for arrays
    and efficient serialization/deserialization.

    The id field contains a unique identifier for the ROI and is managed internally by
    the library. It **MUST** not be set directly by the user.

    Refer to the :ref:`developer guides <extending-guide>` for an example on how to create a new
    ROI class.

    """

    sample: Sample = pydantic.Field(repr=False)
    """The sample where the ROI was extracted from."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_str(cls, ser: str, sample: Sample) -> Self:
        """Create a ROI instance from a JSON string.

        :param ser: a serialized ROI obtained using the `to_str` method
        :param sample: a sample to associate with the ROI
        :return: a new ROI instance.

        """
        return cls(sample=sample, **json.loads(ser))

    def to_str(self) -> str:
        """Serialize a ROI into a string.

        :return: a string serialization of the ROI.

        """
        return self.model_dump_json(exclude={"sample"})


class Annotation(TidyMSBaseModel):
    """Store feature annotation data."""

    roi_id: UUID
    """The ROI id where the feature was extracted from"""

    sample_id: str = ""
    """The sample id where the feature was extracted from"""

    group: int = -1
    """The :term:`feature group` id. Group features from different samples based on their chemical
    identity. Used to create a data matrix. If set to ``-1`` the feature is not assigned to any group.
    """

    isotopologue_label: int = -1
    """Group features from the same isotopic envelope in a sample. If set to ``-1`` the feature is
    not associated with any group of isotopologues in a sample."""

    isotopologue_index: int = -1
    """Position of the feature in an isotopic envelope.  If set to ``-1`` the feature is
    not associated with any group of isotopologues in a sample."""

    charge: int = -1
    """Feature charge state. If set to ``-1`` the feature charge state is not defined"""


RoiType = TypeVar("RoiType", bound=Roi)


class Feature(TidyMSBaseModel, Generic[RoiType]):
    """Base class to represent a feature extracted from a ROI.

    Feature inherits from pydantic BaseModel, and support most of its functionality. New Feature
    subclasses are created by inheritance of this class and setting data fields using Pydantic's
    standard approach.

    There are two field types for features:

    Data fields
        contain information to represent the feature. e.g. the start and end position of a
        chromatographic peak. These fields are represented as standard pydantic fields.
    Descriptors
        describe feature characteristics. e.g, the peak width or peak area in a chromatographic
        peak. **ALL** descriptors **MUST** be floats. These fields are represented as pydantic
        computed fields. Descriptors **MUST** be decorated with `pydantic.computed_field`. It is also
        recommended to use the `functools.cached_property` decorator to cache the descriptor value.
        As an example:

        .. code-block: python

            from functools import cached_property
            from pydantic import computed_field

            class MyFeature(Feature):

                data_field: float = 1.0
                '''A feature data field.'''

                @computed_field
                @cached_property
                def custom_descriptor(self):
                    self.custom_descriptor = 100.0

        The mz, area and height descriptors are set as abstract methods and need
        to be implemented for all concrete Feature classes.

    Finally, three attributes are defined for the Feature class: `id`, `roi` and `annotation`.
    These parameters are managed internally by the library and they MUST never be set directly
    by the user.

    Refer to the :ref:`developer guides <extending-guide>` for an example on how to create a new
    Feature class.

    """

    roi: RoiType = pydantic.Field(repr=False)
    """The ROI where the feature was detected."""

    annotation: Annotation | None = pydantic.Field(repr=False, default=None)
    """Annotation data of the feature."""

    # TODO: remove order functions

    def __lt__(self, other: Self):
        return self.mz < other.mz

    def __le__(self, other: Self):
        return self.mz <= other.mz

    def __gt__(self, other: Self):
        return self.mz > other.mz

    def __ge__(self, other: Self):
        return self.mz >= other.mz

    def __hash__(self) -> int:
        assert self.annotation is not None
        return self.annotation.id.__hash__()

    @cached_property
    @abstractmethod
    def mz(self) -> float:
        """The feature m/z."""
        ...

    @cached_property
    @abstractmethod
    def area(self) -> float:
        """The feature area."""
        ...

    @cached_property
    @abstractmethod
    def height(self) -> float:
        """The feature height."""
        ...

    def has_descriptors_in_range(self, **bounds: tuple[float, float]) -> bool:
        """Check if feature descriptors fall between lower and upper bounds.

        :param bounds: descriptor lower and upper bound values.
        :return: ``True`` if all descriptors fall between the bounds. ``False`` otherwise.

        """
        for name, (lower, upper) in bounds.items():
            if not lower <= self.get(name) <= upper:
                return False
        return True

    def describe(self) -> dict[str, float]:
        """Compute all available descriptors for the feature.

        A descriptor is any method that starts with `get_`.

        :return: a dictionary that maps descriptor names to descriptor values.

        """
        descriptors = dict()
        for descriptor in self.descriptor_names():
            descriptors[descriptor] = self.get(descriptor)
        return descriptors

    def to_str(self) -> str:
        """Serialize the feature data into a string.

        :return: a string serialization of the feature.

        """
        exclude = self.descriptor_names() | {"roi", "annotation"}
        return self.model_dump_json(exclude=exclude)

    @classmethod
    def from_str(cls, s: str, roi: RoiType, annotation: Annotation) -> Self:
        """Create a feature instance from a string.

        :param s: feature string generated with `to_str`.
        :param roi: ROI where the feature was detected.
        :param annotation: the feature annotation.
        :return: a new feature instance.

        """
        d = json.loads(s)
        return cls(roi=roi, annotation=annotation, **d)

    def get(self, descriptor: str) -> float:
        """Compute a descriptor value.

        :param descriptor: the descriptor name.
        :return: the descriptor value.
        :raises ValueError: if an invalid descriptor name is passed.

        """
        try:
            return getattr(self, descriptor)
        except AttributeError as e:
            msg = f"{descriptor} is not a valid descriptor."
            raise ValueError(msg) from e

    @classmethod
    @lru_cache
    def descriptor_names(cls) -> set[str]:
        """Retrieve the available descriptor names.

        :return: the descriptor names.

        """
        return set(cls.model_computed_fields)

    @pydantic.model_validator(mode="after")
    def _add_annotation(self) -> Self:
        if self.annotation is None:
            self.annotation = Annotation(roi_id=self.roi.id, id=self.id, sample_id=self.roi.sample.id)
        return self


class FeatureGroup(pydantic.BaseModel):
    """Store feature group information."""

    group: int = -1
    """the the group id"""

    annotation: GroupAnnotation
    """The feature group annotation."""

    descriptors: dict[str, float]
    """Aggregated feature descriptors."""

    @pydantic.computed_field(repr=False)
    @cached_property
    def mz(self) -> float:
        """The feature m/z."""
        return self.descriptors["mz"]

    def has_descriptors_in_range(self, **filters: tuple[float, float]) -> bool:
        """Check if feature descriptors fall between lower and upper bounds.

        :param bounds: descriptor lower and upper bound values.
        :return: ``True`` if all descriptors fall between the bounds. ``False`` otherwise.

        """
        for name, (lower, upper) in filters.items():
            if name not in self.descriptors:
                msg = f"Feature group {self.group} does not have descriptor {name}."
                raise exceptions.InvalidFeatureDescriptor(msg)
            if not lower <= self.descriptors[name] <= upper:
                return False
        return True


class AnnotableFeature(Feature[RoiType], ABC):
    """Abstract feature class which inherits from Feature.

    Provides extra functionality to perform feature annotation.
    Base feature with also implements methods for feature annotation.

    """

    @abstractmethod
    def compare(self, other: Self) -> float:
        """Compare the similarity between two features.

        Must be a symmetric function that returns a number between 0.0 and 1.0.

        :param other: feature to compare with.
        :return: the similarity between the feature pair.

        """
        ...

    @staticmethod
    @abstractmethod
    def compute_isotopic_envelope(*features) -> IsotopicEnvelope:
        """Compute the isotopic envelope from a list of isotopologue features.

        :param features: the Collection of features used to compute the envelope.
        :return: The normalized isotopic envelope

        """
        ...


class GroupAnnotation(pydantic.BaseModel):
    """Store annotation of a :term:`feature group`."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    label: int
    """The feature group label. Identifies features across samples that are originated from the
    same ionic species."""

    isotopologue_index: int | None = pydantic.Field(default=None, repr=False)
    """The position of the feature in an envelope. Only defined if an isotopologue annotation
    algorithm was applied to the dataset."""

    isotopologue_group: int | None = pydantic.Field(default=None, repr=False)
    """Label shared between isotopologue features. Only defined if an isotopologue annotation
    algorithm was applied to the dataset."""

    envelope: IsotopicEnvelope | None = pydantic.Field(default=None, repr=False)
    """The m/z and abundance values of the envelope members. Only defined if an isotopologue
    annotation algorithm was applied to the dataset and `isotopologue_index` is ``0``."""

    charge: int | None = pydantic.Field(default=None, repr=False)
    """The numerical charge of the feature. Only defined if an isotopologue annotation algorithm
    was applied to the dataset."""

    name: str | None = None
    """An optional name for the feature group."""


class IsotopicEnvelope(pydantic.BaseModel):
    """Store m/z and normalized abundance of a set of isotopic :term:`envelope`."""

    mz: list[float]
    """The envelope sorted m/z"""

    p: list[float]
    """The envelope normalized abundance"""


class MSSpectrum(TidyMSBaseModel):
    """Representation of a Mass Spectrum."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    index: int = -1

    mz: FloatArray1D
    """Sorted m/z data"""

    int: FloatArray1D
    """Spectral intensity"""

    ms_level: pydantic.PositiveInt = 1
    """MS level of the current spectrum"""

    time: pydantic.NonNegativeFloat = 0.0
    """Acquisition time of the spectrum"""

    centroid: bool = True
    """Set to ``True`` if the spectrum was converted to centroid mode. ``False`` otherwise."""

    def get_nbytes(self) -> int:
        """Get the number of bytes stored in m/z and intensity arrays."""
        return self.int.nbytes + self.mz.nbytes


class MZTrace(Roi):
    """ROI implementation using m/z traces.

    An m/z trace is a 1D trace containing m/z, time and intensity information across scans.

    """

    mz: FloatArray1D = pydantic.Field(repr=False)
    """m/z in each scan. All values are assumed to be non-negative."""

    spint: FloatArray1D = pydantic.Field(repr=False)
    """intensity in each scan. All values are assumed to be non-negative."""

    time: FloatArray1D = pydantic.Field(repr=False)
    """time in each scan. All values are assumed to be non-negative."""

    scan: IntArray1D = pydantic.Field(repr=False)
    """scan numbers where the ROI is defined. All values are assumed to be non-negative."""
    noise: FloatArray1D | None = pydantic.Field(repr=False, default=None)
    """if provided, represent the noise level at teach time point."""

    baseline: FloatArray1D | None = pydantic.Field(repr=False, default=None)
    """if provided, represent the baseline level at teach time point."""

    @pydantic.model_validator(mode="after")
    def check_noise_all_non_negative(self) -> Self:
        """Validate noise."""
        if self.noise is not None:
            msg = "All noise array elements must be non-negative."
            assert numpy.all(self.noise >= 0.0), msg
        return self

    @pydantic.model_validator(mode="after")
    def check_baseline_lower_or_equal_than_intensity(self) -> Self:
        """Validate baseline."""
        if self.baseline is not None:
            msg = "All baseline elements must be lower than their corresponding trace intensity."
            assert numpy.all(self.spint >= self.baseline), msg
        return self

    def get_slice_height(self, start: int, end: int) -> FloatArray1D:
        """Compute the trace height in a slice.

        :param start: slice start index.
        :param end: slice end index.

        """
        height = self.spint[start:end]
        if self.baseline is not None:
            height = height - self.baseline[start:end]
        return numpy.maximum(height, 0.0)

    def equals(self, other: Self) -> bool:
        """Check if two m/z traces are equal."""
        return (
            numpy.array_equal(self.mz, other.mz)
            and numpy.array_equal(self.time, other.time)
            and numpy.array_equal(self.spint, other.spint)
            and numpy.array_equal(self.scan, other.scan)
            and self.id == other.id
        )


class Chromatogram(TidyMSBaseModel):
    """Chromatogram representation."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    index: int = -1
    """The chromatogram order in a data file"""

    name: str | None = None
    """The chromatogram name in a data file"""

    time: FloatArray1D
    """The time data"""

    int: FloatArray1D
    """The intensity data"""


class DescriptorPatch(pydantic.BaseModel):
    """Store a descriptor patch."""

    id: UUID
    """The feature id to patch."""

    descriptor: str
    """The descriptor to patch."""

    value: float
    """The new descriptor value to apply"""


class AnnotationPatch(pydantic.BaseModel):
    """Store an annotation patch."""

    id: UUID
    """The feature id to patch."""

    field: str
    """The annotation field fo patch."""

    value: int
    """The new value."""


class FillValue(pydantic.BaseModel):
    """Container class that stores values to fill missing entries in the data matrix."""

    sample_id: str
    """The id of the sample to input the missing value."""

    feature_group: int
    """The feature group to input the missing value."""

    value: float
    """The fill value."""


FeatureType = TypeVar("FeatureType", bound=Feature)
AnnotableFeatureType = TypeVar("AnnotableFeatureType", bound=AnnotableFeature)
