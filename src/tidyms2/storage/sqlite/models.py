"""SQLite assay storage models."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from ...core.dataflow import AssayProcessStatus
from ...core.models import Annotation, AnnotationPatch, DescriptorPatch, Feature, FeatureGroup, FillValue, Roi, Sample

Base = declarative_base()

SAMPLE_TABLE = "samples"
ROI_TABLE = "rois"
FEATURE_TABLE = "features"
ANNOTATION_TABLE = "annotations"
DESCRIPTOR_TABLE = "descriptors"


class BaseOrmModel(Base):
    """Base ORM model."""

    __abstract__ = True

    @classmethod
    def get_unique_columns(cls) -> list[str]:
        """Create a list of field names with unique constraint."""
        return [c.name for c in cls.__table__.c if c.unique]

    def to_dict(self) -> dict[str, Any]:
        """Convert model to a dictionary."""
        return {field.name: getattr(self, field.name) for field in self.__table__.c}

    def update(self, data: dict[str, Any]):
        """Update model using a dictionary."""
        for k, v in data.items():
            if k in self.__table__.c:
                setattr(self, k, v)


class BaseWithId(BaseOrmModel):
    """Base model with id field."""

    __abstract__ = True

    id: Mapped[UUID] = mapped_column(primary_key=True, doc="The record id")


class SampleModel(BaseOrmModel):
    """Stores data from :py:class:`tidyms2.core.models.Sample`."""

    __tablename__ = SAMPLE_TABLE

    id: Mapped[str] = mapped_column(String, primary_key=True)

    path: Mapped[str] = mapped_column(String, nullable=False)
    reader: Mapped[str] = mapped_column(String, nullable=True)
    ms_data_mode: Mapped[str] = mapped_column(String, nullable=True)
    ms_level: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    meta: Mapped[str] = mapped_column(String, nullable=True)

    @classmethod
    def from_sample(cls, sample: Sample) -> SampleModel:
        """Convert to pydantic model."""
        d = sample.model_dump(mode="json", exclude={"meta"})
        d["meta"] = sample.meta.model_dump_json()
        return SampleModel(**d)

    def to_pydantic_model(self) -> Sample:
        """Convert to pydantic model."""
        d = self.to_dict()
        if self.meta is not None:
            d["meta"] = json.loads(d["meta"])
        return Sample(**d)


class RoiModel(BaseWithId):
    """Stores data from :py:class:`tidyms2.core.models.Roi`."""

    __tablename__ = ROI_TABLE

    data: Mapped[str] = mapped_column(String, nullable=False)
    sample_id: Mapped[str] = mapped_column(String, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_roi(cls, roi: Roi, snapshot_id: str) -> RoiModel:
        """Create a new instance from a pydantic model."""
        return cls(id=roi.id, data=roi.to_str(), sample_id=roi.sample.id, snapshot_id=snapshot_id)


class FeatureModel(BaseWithId):
    """Stores data from :py:class:`tidyms2.core.models.Roi`."""

    __tablename__ = FEATURE_TABLE

    data: Mapped[str] = mapped_column(String, nullable=False)
    roi_id: Mapped[UUID] = mapped_column(nullable=False)
    sample_id: Mapped[str] = mapped_column(String, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_feature(cls, feature: Feature, snapshot_id: str) -> FeatureModel:
        """Create a new instance from a pydantic model."""
        return cls(
            id=feature.id,
            data=feature.to_str(),
            roi_id=feature.roi.id,
            sample_id=feature.roi.sample.id,
            snapshot_id=snapshot_id,
        )


class AnnotationModel(BaseWithId):
    """Stores data from :py:class:`tidyms2.core.models.Roi`."""

    __tablename__ = ANNOTATION_TABLE

    roi_id: Mapped[UUID] = mapped_column(nullable=False)
    sample_id: Mapped[str] = mapped_column(String, nullable=False)
    group: Mapped[int] = mapped_column(Integer, nullable=False)
    isotopologue_label: Mapped[int] = mapped_column(Integer, nullable=False)
    isotopologue_index: Mapped[int] = mapped_column(Integer, nullable=False)
    charge: Mapped[int] = mapped_column(Integer, nullable=False)

    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_feature(cls, feature: Feature, snapshot_id: str) -> AnnotationModel:
        """Create a new instance from a pydantic model."""
        annotation: Annotation = feature.annotation  # type: ignore
        return cls(
            id=annotation.id,
            roi_id=annotation.roi_id,
            sample_id=annotation.sample_id,
            group=annotation.group,
            isotopologue_label=annotation.isotopologue_label,
            isotopologue_index=annotation.isotopologue_index,
            charge=annotation.charge,
            snapshot_id=snapshot_id,
        )

    def to_annotation(self) -> Annotation:
        """Convert model to annotation."""
        d = self.to_dict()
        d.pop("snapshot_id")
        return Annotation(**d)


class DescriptorModel(BaseWithId):
    """Stores data from :py:class:`tidyms2.core.models.Roi`."""

    __tablename__ = DESCRIPTOR_TABLE

    data: Mapped[str] = mapped_column(String, nullable=False)

    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_feature(cls, feature: Feature, snapshot_id: str) -> DescriptorModel:
        """Create a new instance from a pydantic model."""
        return cls(id=feature.id, data=json.dumps(feature.describe()), snapshot_id=snapshot_id)

    def create_descriptor_dict(self) -> dict[str, float]:
        """Deserialize model into a dictionary of descriptors."""
        return json.loads(self.data)


class SampleSnapshotModel(BaseOrmModel):
    """Store sample snapshot ids and their order."""

    __tablename__ = "sample_snapshot_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String, nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)


class AssaySnapshotStatusModel(BaseOrmModel):
    """Store assay snapshot ids and their order."""

    __tablename__ = "assay_snapshot_status"

    order: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    def to_status(self) -> AssayProcessStatus:
        """Convert model into an assay process state instance."""
        return AssayProcessStatus.model_validate_json(self.status)


class AnnotationPatchModel(BaseOrmModel):
    """Store patches applied to annotation data."""

    __tablename__ = "annotation_patches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[UUID] = mapped_column(nullable=False)
    field: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_patch(cls, patch: AnnotationPatch, snapshot: str) -> AnnotationPatchModel:
        """Create a new instance from an annotation patch."""
        return AnnotationPatchModel(feature_id=patch.id, field=patch.field, value=patch.value, snapshot_id=snapshot)


class DescriptorPatchModel(BaseOrmModel):
    """Store patches applied to descriptor data."""

    __tablename__ = "descriptor_patches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[UUID] = mapped_column(nullable=False)
    descriptor: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_patch(cls, patch: DescriptorPatch, snapshot: str) -> AnnotationPatchModel:
        """Create a new instance from an annotation patch."""
        return DescriptorPatchModel(
            feature_id=patch.id, descriptor=patch.descriptor, value=patch.value, snapshot_id=snapshot
        )


class FillValueModel(BaseOrmModel):
    """Store patches applied to annotation data."""

    __tablename__ = "fill_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_group: Mapped[int] = mapped_column(nullable=False)
    sample_id: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_fill_value(cls, fill: FillValue, snapshot: str) -> FillValueModel:
        """Create a new instance from an annotation patch."""
        return FillValueModel(
            feature_group=fill.feature_group, sample_id=fill.sample_id, value=fill.value, snapshot_id=snapshot
        )

    def to_fill_value(self) -> FillValue:
        """Create a fill value pydantic model from the current instance."""
        return FillValue(sample_id=self.sample_id, feature_group=self.feature_group, value=self.value)


class FeatureGroupModel(BaseOrmModel):
    """Store feature group data."""

    __tablename__ = "feature_groups"

    group: Mapped[int] = mapped_column(Integer, primary_key=True)
    descriptors: Mapped[str] = mapped_column(nullable=False)
    annotations: Mapped[str] = mapped_column(nullable=False)

    @classmethod
    def from_feature_group(cls, feature_group: FeatureGroup):
        """Create a new model from a feature group."""
        return cls(
            group=feature_group.group,
            descriptors=json.dumps(feature_group.descriptors),
            annotations=feature_group.annotation.model_dump_json(),
        )

    def to_feature_group(self) -> FeatureGroup:
        """Convert a model into a feature group."""
        return FeatureGroup.model_validate(
            {
                "group": self.group,
                "annotation": json.loads(self.annotations),
                "descriptors": json.loads(self.descriptors),
            }
        )
