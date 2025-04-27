"""Storage interface for sample data and assay data."""

from __future__ import annotations

from itertools import product
from typing import Generic, Iterable, Protocol, TypeVar
from uuid import UUID

from .dataflow import AssayProcessStatus, SampleProcessStatus
from .models import Annotation, AnnotationPatch, DescriptorPatch, FeatureGroup, FeatureType, FillValue, RoiType, Sample


class SampleStorage(Protocol, Generic[RoiType, FeatureType]):
    """Interface to store sample data.

    Provides:

    Access sample metadata.
        Sample metadata maintains all sample-related information. It also provides utilities
        to access raw data.
    add/get/delete ROIs and features.
        Provide utilities to modify the sample data.
    sample data status
        `get_status` retrieves a record of the information available for the sample,
        such as if ROI extraction was performed on the data, feature extraction,
        isotopologue annotation, etc... `set_status` allow to specify this status.
    linear data snapshots
        Sample storage maintains a list of snapshots with independent copies of the data
        at different points in time. The `create_snapshot` allows to create data snapshots
        of the sample data at any point in time. Only the latest snapshot is modifiable.
        The `reset` method rollbacks the data to a specified snapshot in read only mode,
        i.e., all operations that modify the snapshot state should raise SnapshotError.
        The ``hard`` parameter controls if a hard reset operation is performed on the data.
        That is, setting the selected snapshot as the latest and deleting data from all
        posterior snapshots. Finally, the `set_latest_snapshot` set the data to the latest
        snapshot.

    """

    roi_type: type[RoiType]
    feature_type: type[FeatureType]

    def __init__(self, sample: Sample, roi_type: type[RoiType], feature_type: type[FeatureType], **kwargs) -> None: ...

    def add_features(self, *features: FeatureType) -> None:
        """Add features to the sample storage."""
        ...

    def add_rois(self, *rois: RoiType) -> None:
        """Add ROIs to the sample storage."""
        ...

    def create_snapshot(self, snapshot_id: str) -> None:
        """Create a snapshot of the sample data."""
        ...

    def delete_features(self, *feature_ids: UUID) -> None:
        """Delete features from the sample storage using their ids."""
        ...

    def delete_rois(self, *roi_id: UUID) -> None:
        """Delete ROIs from the sample storage using their ids."""
        ...

    def get_feature(self, feature_id: UUID) -> FeatureType:
        """Get a feature from the sample storage using its id."""
        ...

    def get_n_features(self) -> int:
        """Get the total number of features in the storage."""
        ...

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the storage."""
        ...

    def get_roi(self, roi_id: UUID) -> RoiType:
        """Get a ROI from the sample storage using its id."""
        ...

    def get_sample(self) -> Sample:
        """Get the sample metadata."""
        ...

    def get_snapshot_id(self) -> str:
        """Get the current snapshot id."""
        ...

    def get_status(self) -> SampleProcessStatus:
        """Get the current process status."""
        ...

    def set_status(self, status: SampleProcessStatus) -> None:
        """Set the current process status."""
        ...

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature with the provided id exists."""
        ...

    def has_roi(self, roi_id: UUID) -> bool:
        """Check if a ROI with the provided id exists."""
        ...

    def list_features(self, roi_id: UUID | None = None) -> list[FeatureType]:
        """List features extracted from the sample."""
        ...

    def list_rois(self) -> list[RoiType]:
        """List ROIs extracted from the sample."""
        ...

    def list_snapshots(self) -> list[str]:
        """List all available snapshots."""
        ...

    def set_snapshot(self, snapshot_id: str | None = None, reset: bool = False) -> None:
        """Set the sample data to the specified snapshot id.

        Setting `reset` to ``True`` set the selected status as latest and deletes all posterior snapshots.

        """
        ...


class AssayStorage(Protocol, Generic[RoiType, FeatureType]):
    """Interface to store assay data."""

    def add_feature_groups(self, *groups: FeatureGroup) -> None:
        """Add feature groups data to the assay storage."""
        ...

    def add_fill_values(self, *fill_values: FillValue) -> None:
        """Add values to fill missing entries in the data matrix."""
        ...

    def add_sample_data(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        """Add data from a sample.

        If include snapshots is set to ``True`` include all data snapshots. Otherwise, include only the
        latest data snapshot.

        """
        ...

    def create_snapshot(self, snapshot_id: str) -> None:
        """Create a new assay data snapshot."""
        ...

    def fetch_annotations(self, sample_id: str | None = None) -> list[Annotation]:
        """Fetch the feature annotations."""
        ...

    def fetch_descriptors(self, sample_id: str | None = None, descriptors: Iterable[str] | None = None) -> dict:
        """Fetch the feature descriptors."""
        ...

    def fetch_features_by_group(self, group: int) -> list[FeatureType]:
        """Fetch features using the feature group id."""
        ...

    def fetch_features_by_id(self, *feature_ids: UUID) -> list[FeatureType]:
        """Fetch features using their ids."""
        ...

    def fetch_features_by_sample(self, sample_id: str) -> list[FeatureType]:
        """Fetch features using the sample id."""
        ...

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups stored in the assay."""
        ...

    def fetch_fill_values(self) -> dict[str, dict[int, float]]:
        """Fetch fill values for data matrix."""
        ...

    def fetch_rois_by_id(self, *roi_ids: UUID) -> list[RoiType]:
        """Fetch ROIs using their ids."""
        ...

    def fetch_rois_by_sample(self, sample_id: str) -> list[RoiType]:
        """Fetch all ROIs from a sample."""
        ...

    def fetch_sample(self, sample_id: str) -> Sample:
        """Fetch sample metadata from a sample using its id."""
        ...

    def fetch_sample_data(self, sample_id: str) -> SampleStorage[RoiType, FeatureType]:
        """Fetch Samples from the assay using their ids."""
        ...

    def get_n_features(self) -> int:
        """Get the total number of features in the assay."""
        ...

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the assay."""
        ...

    def get_process_status(self) -> AssayProcessStatus:
        """Get the current process status."""
        ...

    def get_snapshot_id(self) -> str:
        """Retrieve the current snapshot id."""
        ...

    def get_feature_type(self) -> type[FeatureType]:
        """Retrieve the Feature class used."""
        ...

    def get_roi_type(self) -> type[RoiType]:
        """Retrieve the ROI class used."""
        ...

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature with the provided id exists."""
        ...

    def has_feature_group(self, feature_group: int) -> bool:
        """Check if a group with the provided id is in the assay."""
        ...

    def has_roi(self, roi_id: UUID) -> bool:
        """Check if a ROI with the provided id exists."""
        ...

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample with the provided id is in the assay."""
        ...

    def list_feature_groups(self) -> list[int]:
        """List all group ids in the assay."""
        ...

    def list_samples(self) -> list[Sample]:
        """Fetch metadata from all samples in the assay."""
        ...

    def list_snapshots(self) -> list[str]:
        """Retrieve the list of all snapshots."""
        ...

    def patch_annotations(self, *patches: AnnotationPatch) -> None:
        """Update feature annotation values."""
        ...

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Update feature descriptors values."""
        ...

    def set_process_status(self, status: AssayProcessStatus) -> None:
        """Set the new process status."""
        ...

    def set_snapshot(self, snapshot_id: str | None = None, reset: bool = False) -> None:
        """Set assay storage data to specified snapshot.

        If ``None``, fetch data from the latest snapshot.

        """
        ...


def find_missing_entries(assay_storage: AssayStorage) -> dict[str, list[int]]:
    """List all data matrix missing entries from data in an assay.

    Entries are sorted by sample_id and feature group.

    """
    sample_ids = [x.id for x in assay_storage.list_samples()]
    group_ids = assay_storage.list_feature_groups()
    all_entries = set(product(sample_ids, group_ids))
    existing_entries = {(x.sample_id, x.group) for x in assay_storage.fetch_annotations() if x.group > -1}
    missing_entries = dict()

    for sample, group in all_entries.difference(existing_entries):
        sample_list = missing_entries.setdefault(sample, list())
        sample_list.append(group)
    return missing_entries


def delete_empty_rois(sample_storage: SampleStorage[RoiType, FeatureType]) -> None:
    """Remove ROI that do not contain any features from the sample storage.

    :param sample_storage: any instance that implements the SampleStorage protocol.

    """
    roi_with_feature_ids = {x.roi.id for x in sample_storage.list_features()}
    all_roi_ids = {x.id for x in sample_storage.list_rois()}
    empty_roi_ids = all_roi_ids.difference(roi_with_feature_ids)
    sample_storage.delete_rois(*empty_roi_ids)


_SAMPLE_STORAGES: dict[str, SampleStorage] = dict()


T = TypeVar("T")


def register_sample_storage(storage_type: type[T]) -> type[T]:
    """Register a sample storage.

    :param storage_type: A class implementing the SampleStorage protocol to register

    """
    if storage_type.__name__ in _SAMPLE_STORAGES:
        raise ValueError
    return storage_type
