"""In memory sample data storage implementation."""

from __future__ import annotations

from typing import Generic, Iterable
from uuid import UUID

from ..core import exceptions, storage
from ..core.dataflow import AssayProcessStatus, SampleProcessStatus
from ..core.models import (
    Annotation,
    AnnotationPatch,
    DescriptorPatch,
    FeatureGroup,
    FeatureType,
    FillValue,
    RoiType,
    Sample,
)
from ..core.storage import SampleStorage
from ..core.utils.common import create_id
from .table import FeatureTable

LATEST = "head"


class OnMemoryAssayStorage(Generic[RoiType, FeatureType]):
    """Store assay data in memory."""

    def __init__(self, id: str, roi_type: type[RoiType], feature_type: type[FeatureType]) -> None:
        self.id = id

        self._feature_type = feature_type
        self._roi_type = roi_type

        self._sample_data: dict[str, OnMemorySampleStorage[RoiType, FeatureType]] = dict()
        self._rois_to_sample_id: dict[UUID, str] = dict()

        self._current = OnMemoryAssayStorageSnapshot(LATEST, AssayProcessStatus())
        self._snapshots = [self._current]

    def add_sample_data(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        """Add samples to the assay."""
        sample = data.get_sample()
        if self.has_sample(sample.id):
            raise exceptions.RepeatedIdError(sample.id)

        copy = OnMemorySampleStorage.from_sample_storage(data)

        new_rois = dict()
        sample_id = copy.get_sample().id
        new_rois.update({x.id: sample_id for x in copy.list_rois()})

        new_features = dict()
        sample_id = copy.get_sample().id
        features = copy.list_features()
        new_features.update({x.id: sample_id for x in features})

        descriptors = list()
        annotations = list()
        for ft in features:
            descriptors.append(ft.describe())
            annotations.append(ft.annotation)

        annotations = [x.annotation for x in features]
        self._current.add_descriptors(descriptors, annotations)

        self._rois_to_sample_id.update(new_rois)
        self._sample_data[sample.id] = copy
        self._groups = list()

    def add_fill_values(self, *fill_values: FillValue) -> None:
        """Add values to fill missing data matrix entries."""
        d = dict()
        for fill in fill_values:
            sample_d = d.setdefault(fill.sample_id, dict())
            sample_d[fill.feature_group] = fill.value
        self._current.add_fill_values(d)

    def add_feature_groups(self, *feature_groups: FeatureGroup) -> None:
        """Add feature groups to the assay."""
        self._current.add_feature_groups(*feature_groups)

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups from the assay."""
        return self._current.fetch_feature_groups()

    def create_snapshot(self, snapshot_id) -> None:
        """Create a new sample data snapshot.

        :param snapshot_id: the id for the new snapshot.
        :raises RepeatedIdError: if a snapshot with this id already exists.

        """
        self._check_latest_snapshot()
        if snapshot_id == LATEST:
            raise ValueError(f"snapshot id `{LATEST}` is reserved for internal use.")

        if snapshot_id in self.list_snapshots():
            msg = f"Snapshot with id={snapshot_id} already exists for assay {self.id}."
            raise exceptions.RepeatedIdError(msg)

        self._current.id = snapshot_id
        latest = self._current.copy_snapshot(LATEST)
        self._snapshots.append(latest)
        self._current = latest

    def fetch_fill_values(self) -> dict[str, dict[int, float]]:
        """Fetch fill values for missing data matrix entries."""
        return self._current.fetch_fill_values(copy=True)

    def get_process_status(self) -> AssayProcessStatus:
        """Retrieve the current process status."""
        return self._current.status.model_copy(deep=True)

    def set_process_status(self, status: AssayProcessStatus) -> None:
        """Set the assay process status."""
        self._current.status = status

    def fetch_sample_data(self, sample_id: str) -> OnMemorySampleStorage[RoiType, FeatureType]:
        """Fetch Samples from the assay using their ids."""
        if not self.has_sample(sample_id):
            raise exceptions.SampleNotFound(sample_id)
        return self._sample_data[sample_id]

    def fetch_sample(self, sample_id: str) -> Sample:
        """Fetch Samples from the assay using their ids."""
        return self.fetch_sample_data(sample_id).get_sample()

    def has_sample(self, sample_id: str) -> bool:
        """Check if the assay contains a sample with the provided id."""
        return sample_id in self._sample_data

    def list_samples(self) -> list[Sample]:
        """Fetch all samples in the assay."""
        return [x.get_sample().model_copy(deep=True) for x in self._sample_data.values()]

    def list_snapshots(self) -> list[str]:
        """List all snapshot ids."""
        return [x.id for x in self._snapshots]

    def list_feature_groups(self) -> list[int]:
        """List all feature groups in the assay."""
        return self._current.list_feature_groups()

    def get_feature_type(self) -> type[FeatureType]:
        """Retrieve the Feature class used."""
        return self._feature_type

    def get_roi_type(self) -> type[RoiType]:
        """Retrieve the ROI class used."""
        return self._roi_type

    def get_n_features(self) -> int:
        """Get the total number of features in the assay."""
        return sum(x.get_n_features() for x in self._sample_data.values())

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the assay."""
        return sum(x.get_n_rois() for x in self._sample_data.values())

    def get_snapshot_id(self) -> str:
        """Get the current snapshot id."""
        return self._current.id

    def has_roi(self, roi_id: UUID) -> bool:
        """Check if a ROI is in the storage."""
        return roi_id in self._rois_to_sample_id

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a Feature with the provided id is in the storage."""
        return self._current.has_feature(feature_id)

    def has_feature_group(self, feature_group: int) -> bool:
        """Check if a group with the provided id is in the assay."""
        return self._current.has_feature_group(feature_group)

    def fetch_annotations(self, sample_id: str | None = None) -> list[Annotation]:
        """Fetch a copy of the feature annotations.

        :param sample_id: If provided, only fetch annotations from this sample. By default, fetch annotations
            from all samples.
        :raises SampleNotFound: if a sample id that is not in the assay storage is provided.

        """
        if sample_id is not None and not self.has_sample(sample_id):
            raise exceptions.SampleNotFound(sample_id)

        return self._current.fetch_annotations(sample_id=sample_id, copy=True)

    def fetch_descriptors(
        self, sample_id: str | None = None, descriptors: Iterable[str] | None = None
    ) -> dict[str, list[float]]:
        """Fetch a copy of the feature descriptors.

        :param sample_id: If provided, only fetch descriptors from this sample. Otherwise, fetch descriptors
            from all samples
        :param descriptors: If provided only fetch values from these descriptors. By default, all descriptors
            are fetched.
        :raises SampleNotFound: if a sample id that is not in the assay storage is provided.
        :raises InvalidFeatureDescriptor: If an undefined descriptor name for the assay feature type is provided.
        """
        if sample_id is not None and not self.has_sample(sample_id):
            raise exceptions.SampleNotFound(sample_id)

        all_descriptors = self._feature_type.descriptor_names()
        if descriptors is None:
            descriptors = list(all_descriptors)

        for d in descriptors:
            if d not in all_descriptors:
                msg = f"{d} is not a valid descriptor of {self._feature_type.__name__}."
                raise exceptions.InvalidFeatureDescriptor(msg)

        return self._current.fetch_descriptors(descriptors=descriptors, sample_id=sample_id, copy=True)

    def fetch_rois_by_sample(self, sample_id: str) -> list[RoiType]:
        """Retrieve ROIs from the storage."""
        return [x.model_copy(deep=True) for x in self.fetch_sample_data(sample_id).list_rois()]

    def fetch_rois_by_id(self, *roi_ids: UUID) -> list[RoiType]:
        """Fetch a ROI using its id."""
        roi_list = list()
        for id_ in roi_ids:
            if not self.has_roi(id_):
                raise exceptions.RoiNotFound(id_)

            sample_id = self._rois_to_sample_id[id_]
            roi = self._sample_data[sample_id].get_roi(id_).model_copy(deep=True)
            roi_list.append(roi)
        return roi_list

    def fetch_features_by_id(self, *feature_ids: UUID) -> list[FeatureType]:
        """Fetch a feature using its id."""
        # first we list all features by id
        feature_list = list()
        for id_ in feature_ids:
            if not self.has_feature(id_):
                raise exceptions.FeatureNotFound(id_)
            sample_id = self._current.get_sample_id(id_)
            ft = self._sample_data[sample_id].get_feature(id_)
            feature_list.append(ft)

        # multiple features may be extracted from the same ROI, we create a copy of each
        # unique ROI to avoid duplicates
        unique_roi_ids = {x.roi.id for x in feature_list}
        roi_id_to_roi = {x.id: x for x in self.fetch_rois_by_id(*unique_roi_ids)}

        # finally we create copy of all features but pass the same ROI copy to features
        # that share ROIs
        return [ft.model_copy(deep=True, update={"roi": roi_id_to_roi[ft.roi.id]}) for ft in feature_list]

    def fetch_features_by_sample(self, sample_id: str) -> list[FeatureType]:
        """Retrieve all features from a sample."""
        # here we also have the problem of shared ROIs as in the fetch_features_by_id method
        # so we use the same approach
        all_ids = [x.id for x in self.fetch_sample_data(sample_id).list_features()]
        return self.fetch_features_by_id(*all_ids)

    def fetch_features_by_group(self, group: int) -> list[FeatureType]:
        """Retrieve all features belonging to a feature group."""
        return self.fetch_features_by_id(*self._current.get_ids_by_group(group))

    def patch_annotations(self, *patches: AnnotationPatch) -> None:
        """Update feature annotation values."""
        self._check_latest_snapshot()
        self._current.patch_annotation(*patches)

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Update feature descriptor values."""
        self._check_latest_snapshot()
        self._current.patch_descriptors(*patches)

    def set_snapshot(self, snapshot_id: str | None = None, reset: bool = False) -> None:
        """Set snapshot from which the storage will fetch data from.

        :param snapshot_id: the snapshot to set
        :param reset: set the selected snapshot as the latest and delete posterior snapshots.
            Note that the selected snapshot id will be set to `head`.
        :raises SnapshotNotFoundError: if the provided `snapshot_id` is not in the storage

        """
        if snapshot_id is None:
            snapshot_id = LATEST
        try:
            snapshot_index = self.list_snapshots().index(snapshot_id)
            self._current = self._snapshots[snapshot_index]
        except ValueError as e:
            msg = f"Snapshot {snapshot_id} not found in {self.id} assay storage."
            raise exceptions.SnapshotNotFound(msg) from e

        if reset:
            self._snapshots = self._snapshots[:snapshot_index]
            self._current.id = LATEST

    def _check_latest_snapshot(self):
        if self._current.id != LATEST:
            msg = f"Only latest snapshot can be modified. Cannot change the state of snapshot {self._current.id}"
            raise exceptions.SnapshotError(msg)


@storage.register_sample_storage
class OnMemorySampleStorage(Generic[RoiType, FeatureType]):
    """Store sample data in memory.

    Manages accession to sample and ROIs in O(1) time. Both add features and
    add ROI operations are atomic and consistent operations.

    """

    def __init__(self, sample: Sample, roi_type: type[RoiType], feature_type: type[FeatureType], **kwargs) -> None:
        self.roi_type = roi_type
        self.feature_type = feature_type
        self._sample = sample.model_copy()
        self._snapshots = list()
        self._current: None | OnMemorySampleStorageSnapshot = None

    def _add_snapshot(self, snapshot: OnMemorySampleStorageSnapshot[RoiType, FeatureType]):
        self._snapshots.append(snapshot)

    def _get_current_snapshot(self) -> OnMemorySampleStorageSnapshot[RoiType, FeatureType]:
        # create a snapshot if it does not exist
        if self._current is None:
            state = SampleProcessStatus()
            latest = OnMemorySampleStorageSnapshot(self._sample, LATEST, state, self.roi_type, self.feature_type)
            self._current = latest
            self._snapshots.append(latest)
        return self._current

    def add_features(self, *features: FeatureType) -> None:
        """Add features to the sample storage.

        :param features: the features to be add.
        :raises RepeatedIdError: if a feature with an existing id is provided.
        :raises RoiNotFoundError: if trying to add a feature associated with a ROI not in the storage

        """
        self._check_latest_snapshot()
        self._get_current_snapshot().add_features(*features)

    def add_rois(self, *rois: RoiType) -> None:
        """Add ROIs to the sample storage.

        :param rois: the rois to be add
        :raises RepeatedIdError: if a ROI with this id already exists.

        """
        self._check_latest_snapshot()
        self._get_current_snapshot().add_rois(*rois)

    def create_snapshot(self, snapshot_id: str) -> None:
        """Create a new sample data snapshot.

        :param snapshot_id: the id for the new snapshot.
        :raises RepeatedIdError: if a snapshot with this id already exists.

        """
        self._check_latest_snapshot()
        if snapshot_id == LATEST:
            raise ValueError(f"snapshot id `{LATEST}` is reserved for internal use.")

        if snapshot_id in self.list_snapshots():
            msg = f"Snapshot with id={snapshot_id} already exists for sample {self._sample.id}."
            raise exceptions.RepeatedIdError(msg)

        self._snapshots[-1].id = snapshot_id

        latest = self._get_current_snapshot().copy(LATEST, set_new_ids=True)

        self._snapshots.append(latest)
        self._current = latest

    def delete_features(self, *feature_ids: UUID) -> None:
        """Delete features using their ids.

        Non-existing ids are ignored.

        """
        self._check_latest_snapshot()
        self._get_current_snapshot().delete_features(*feature_ids)

    def delete_rois(self, *roi_ids: UUID) -> None:
        """Delete ROIs using their ids.

        Non-existing ids are ignored.

        """
        self._check_latest_snapshot()
        self._get_current_snapshot().delete_rois(*roi_ids)

    def get_feature(self, feature_id: UUID) -> FeatureType:
        """Retrieve a feature by id.

        :raises FeatureNotFoundError: if the provided `feature_id` is not in the storage

        """
        return self._get_current_snapshot().get_feature(feature_id)

    def get_n_features(self) -> int:
        """Get the total number of features in the storage."""
        return self._get_current_snapshot().get_n_features()

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the storage."""
        return self._get_current_snapshot().get_n_rois()

    def get_roi(self, roi_id: UUID) -> RoiType:
        """Retrieve a ROI by id.

        :raises RoiNotFoundError: if the provided `roi_id` is not in the storage

        """
        return self._get_current_snapshot().get_roi(roi_id)

    def get_sample(self) -> Sample:
        """Retrieve the storage sample."""
        return self._sample

    def get_snapshot_id(self) -> str:
        """Get the current snapshot id."""
        return self._get_current_snapshot().id

    def get_status(self) -> SampleProcessStatus:
        """Get the current process status."""
        return self._get_current_snapshot().status

    def has_feature(self, feature_id: UUID) -> bool:
        """Check the existence of a feature using its id."""
        return self._get_current_snapshot().has_feature(feature_id)

    def has_roi(self, roi_id: UUID) -> bool:
        """Check the existence of a ROI with the specified id."""
        return self._get_current_snapshot().has_roi(roi_id)

    def list_features(self, roi_id: UUID | None = None) -> list[FeatureType]:
        """List stored features.

        :param roi_id: if provided, only features associated with this ROI are listed
        :raises RoiNotFoundError: if the provided `roi_id` is not in the storage

        """
        return self._get_current_snapshot().list_features(roi_id)

    def list_rois(self) -> list[RoiType]:
        """List all stored ROIs."""
        return self._get_current_snapshot().list_rois()

    def list_snapshots(self) -> list[str]:
        """List all snapshots."""
        return [x.id for x in self._snapshots]

    def set_snapshot(self, snapshot_id: str | None = None, reset: bool = False) -> None:
        """Set snapshot from which the storage will fetch data from.

        :param snapshot_id: the snapshot to set
        :param reset: set the selected snapshot as the latest and delete posterior snapshots.
            The selected snapshot id will be set to `head`.
        :raises SnapshotNotFoundError: if the provided `snapshot_id` is not in the storage

        """
        if snapshot_id is None:
            snapshot_id = LATEST
        try:
            snapshot_index = self.list_snapshots().index(snapshot_id)
            self._current = self._snapshots[snapshot_index]
        except ValueError as e:
            msg = f"Snapshot {snapshot_id} not found in {self._sample.id} storage."
            raise exceptions.SnapshotNotFound(msg) from e

        if reset:
            self._snapshots = self._snapshots[:snapshot_index]
            self._get_current_snapshot().id = LATEST

    def set_status(self, status: SampleProcessStatus) -> None:
        """Set the current process status."""
        self._check_latest_snapshot()
        self._get_current_snapshot().status = status

    def _check_latest_snapshot(self):
        id_ = self._get_current_snapshot().id
        if id_ != LATEST:
            msg = f"Only latest snapshot can be modified. Cannot change the state of snapshot {id_}"
            raise exceptions.SnapshotError(msg)

    @classmethod
    def from_sample_storage(
        cls, sample_storage: SampleStorage[RoiType, FeatureType]
    ) -> OnMemorySampleStorage[RoiType, FeatureType]:
        """Create a new instance using the provided sample storage."""
        sample = sample_storage.get_sample()
        copied = OnMemorySampleStorage(sample, sample_storage.roi_type, sample_storage.feature_type)

        current_snapshot_id = sample_storage.get_snapshot_id()

        for snapshot_id in sample_storage.list_snapshots():
            sample_storage.set_snapshot(snapshot_id)
            snapshot = OnMemorySampleStorageSnapshot.from_sample_storage(sample_storage)
            copied._add_snapshot(snapshot)
        copied.set_snapshot(LATEST)

        sample_storage.set_snapshot(current_snapshot_id)
        return copied

    @classmethod
    def from_dict(
        cls,
        sample: Sample,
        rois: dict[str, list[RoiType]],
        features: dict[str, list[FeatureType]],
        snapshots: list[str],
        states: dict[str, SampleProcessStatus],
        roi_type: type[RoiType],
        feature_type: type[FeatureType],
    ) -> OnMemorySampleStorage[RoiType, FeatureType]:
        """Create a new instance from sample, Roi and Feature data.

        :param sample: the sample associated with the data
        :param rois: a dictionary that maps snapshot ids to a list of ROIs in the snapshot
        :param features: a dictionary that maps snapshot ids to a list of features in the snapshot
        :param snapshots: the list of snapshots to create
        :param state_list: a mapping from snapshot id to sample data state

        """
        data = cls(sample, roi_type, feature_type)
        for snapshot_id in snapshots:
            snapshot_roi = rois.get(snapshot_id)
            if snapshot_roi is None:
                raise ValueError(f"rois dictionary must contain a ROI list for snapshot {snapshot_id}.")
            snapshot_features = features.get(snapshot_id)

            if snapshot_features is None:
                raise ValueError(f"rois dictionary must contain a ROI list for snapshot {snapshot_id}.")

            snapshot_state = states.get(snapshot_id)
            if snapshot_state is None:
                raise ValueError(f"states dictionary must contain a state for snapshot {snapshot_id}.")

            snapshot = OnMemorySampleStorageSnapshot(sample, snapshot_id, snapshot_state, roi_type, feature_type)
            snapshot.add_rois(*snapshot_roi)
            snapshot.add_features(*snapshot_features)
            data._add_snapshot(snapshot)
        return data


class OnMemorySampleStorageSnapshot(Generic[RoiType, FeatureType]):
    """Stores data state during a sample processing pipeline."""

    def __init__(
        self,
        sample: Sample,
        snapshot_id: str,
        status: SampleProcessStatus,
        roi_type: type[RoiType],
        feature_type: type[FeatureType],
    ):
        self._roi_type = roi_type
        self._feature_type = feature_type
        self.sample = sample
        self.id = snapshot_id
        self._features: dict[UUID, FeatureType] = dict()
        self._rois: dict[UUID, RoiType] = dict()
        self._roi_to_features: dict[UUID, set[UUID]] = dict()
        self.status = status

    def copy(self, snapshot_id: str, set_new_ids: bool = False) -> OnMemorySampleStorageSnapshot[RoiType, FeatureType]:
        """Create a copy of the snapshot."""
        status = self.status.model_copy()
        copy = OnMemorySampleStorageSnapshot(self.sample, snapshot_id, status, self._roi_type, self._feature_type)

        rois_copy = list()
        roi_map = dict()
        for roi in self.list_rois():
            roi_copy = roi.model_copy(deep=True)
            rois_copy.append(roi_copy)
            if set_new_ids:
                roi_copy.id = create_id()
            roi_map[roi.id] = roi_copy

        features_copy = list()
        for ft in self.list_features():
            ft_copy = ft.model_copy(deep=True, update={"roi": roi_map[ft.roi.id]})
            features_copy.append(ft_copy)
            if set_new_ids:
                ft_copy.id = create_id()
                ft_copy.annotation.id = ft_copy.id  # type: ignore

        copy.add_rois(*rois_copy)
        copy.add_features(*features_copy)
        return copy

    def add_features(self, *features: FeatureType) -> None:
        """Add features to the snapshot."""
        new_features = dict()
        for ft in features:
            if not self.has_roi(ft.roi.id):
                msg = f"Cannot add feature {ft.id} because its parent ROI {ft.roi.id} was not found in the data."
                raise exceptions.RoiNotFound(msg)

            if self.has_feature(ft.id):
                msg = f"A feature with id={ft.id} already exists in {self.sample.id}/{self.id}"
                raise exceptions.RepeatedIdError(msg)

            new_features[ft.id] = ft
        self._features.update(new_features)

        for ft_id, ft in new_features.items():
            self._roi_to_features[ft.roi.id].add(ft_id)

    def add_rois(self, *rois: RoiType) -> None:
        """Add Rois to the snapshot."""
        new_rois = dict()
        new_roi_to_features = dict()
        for roi in rois:
            if self.has_roi(roi.id):
                msg = f"ROI with id {roi.id} already stored in {self.sample.id}/{self.id}."
                raise exceptions.RepeatedIdError(msg)
            new_rois[roi.id] = roi
            new_roi_to_features[roi.id] = set()
        self._rois.update(new_rois)
        self._roi_to_features.update(new_roi_to_features)

    def delete_features(self, *feature_ids: UUID) -> None:
        """Delete features from the snapshot."""
        for id_ in feature_ids:
            if not self.has_feature(id_):
                continue
            ft = self._features.pop(id_)
            parent_roi_features = self._roi_to_features[ft.roi.id]
            parent_roi_features.remove(id_)

    def delete_rois(self, *roi_ids: UUID) -> None:
        """Delete ROIs and their associated features."""
        for id_ in roi_ids:
            if not self.has_roi(id_):
                continue

            del self._rois[id_]

            features = self._roi_to_features.pop(id_)
            for child_id in features:
                del self._features[child_id]

    def get_feature(self, feature_id: UUID) -> FeatureType:
        """Retrieve a feature by id."""
        feature = self._features.get(feature_id)
        if feature is None:
            msg = f"Feature {feature_id} not found  in {self.sample.id}/{self.id}."
            raise exceptions.FeatureNotFound(msg)
        return feature

    def get_roi(self, roi_id: UUID) -> RoiType:
        """Retrieve a ROI by id."""
        roi = self._rois.get(roi_id)
        if roi is None:
            msg = f"ROI {roi_id} not found in {self.sample.id}/{self.id}."
            raise exceptions.RoiNotFound(msg)
        return roi

    def get_n_features(self) -> int:
        """Get the total number of features in the snapshot."""
        return len(self._features)

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the snapshot."""
        return len(self._rois)

    def has_feature(self, feature_id: UUID) -> bool:
        """CHeck the existence of a feature using its id."""
        return feature_id in self._features

    def has_roi(self, roi_id: UUID) -> bool:
        """Check the existence of a ROI."""
        return roi_id in self._rois

    def list_rois(self) -> list[RoiType]:
        """List all ROIs in the sample storage."""
        return list(self._rois.values())

    def list_features(self, roi_id: UUID | None = None) -> list[FeatureType]:
        """List all features in the snapshot."""
        if roi_id is None:
            return list(self._features.values())

        if not self.has_roi(roi_id):
            msg = f"Roi {roi_id} not found in {self.sample.id}/{self.id}."
            raise exceptions.RoiNotFound(msg)

        return [self.get_feature(x) for x in self._roi_to_features[roi_id]]

    @classmethod
    def from_sample_storage(
        cls, sample_storage: SampleStorage[RoiType, FeatureType]
    ) -> OnMemorySampleStorageSnapshot[RoiType, FeatureType]:
        """Create a snapshot from sample storage current status."""
        sample = sample_storage.get_sample()
        status = sample_storage.get_status()
        snapshot_id = sample_storage.get_snapshot_id()
        snapshot = cls(sample, snapshot_id, status, sample_storage.roi_type, sample_storage.feature_type)
        snapshot.add_rois(*sample_storage.list_rois())
        snapshot.add_features(*sample_storage.list_features())
        return snapshot.copy(snapshot_id, set_new_ids=False)


class OnMemoryAssayStorageSnapshot:
    """Store independent copies of assay descriptors and missing values."""

    def __init__(self, snapshot_id: str, status: AssayProcessStatus):
        self.id = snapshot_id
        self.status = status
        self._table = FeatureTable()

    def add_descriptors(self, descriptors: list[dict[str, float]], annotations: list[Annotation]):
        """Add annotations and descriptors from a sample."""
        self._table.add_descriptors(descriptors, annotations)

    def add_feature_groups(self, *feature_groups: FeatureGroup) -> None:
        """Add feature groups to the snapshot."""
        self._table.add_feature_groups(*feature_groups)

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups from the snapshot."""
        return self._table.fetch_feature_groups()

    def has_feature_group(self, group: int) -> bool:
        """Check if a group with the provided id is stored in the assay."""
        return self._table.has_feature_group(group)

    def fetch_annotations(self, sample_id: str | None = None, copy: bool = False) -> list[Annotation]:
        """Create a list feature annotations.

        :param sample_id: If provided, only include annotations from this sample
        """
        return self._table.fetch_annotations(sample_id=sample_id, copy=copy)

    def fetch_descriptors(
        self, descriptors: Iterable[str] | None = None, sample_id: str | None = None, copy: bool = False
    ) -> dict[str, list[float]]:
        """Fetch descriptors from the snapshot."""
        return self._table.fetch_descriptors(descriptors=descriptors, sample_id=sample_id, copy=copy)

    def fetch_fill_values(self, copy: bool = False) -> dict[str, dict[int, float]]:
        """Fetch snapshot fill values."""
        return self._table.fetch_fill_values(copy=copy)

    def add_fill_values(self, fill_values: dict[str, dict[int, float]]) -> None:
        """Add missing values to the snapshot."""
        self._table.add_fill_values(fill_values)

    def get_sample_id(self, feature_id: UUID) -> str:
        """Retrieve the sample id of a feature."""
        return self._table.get_sample_id(feature_id)

    def get_ids_by_group(self, group: int) -> list[UUID]:
        """Retrieve all feature ids associated with a feature group."""
        return self._table.get_ids_by_group(group)

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature is in the snapshot."""
        return self._table.has_feature(feature_id)

    def patch_annotation(self, *patches: AnnotationPatch) -> None:
        """Apply patches to annotations."""
        self._table.patch_annotation(*patches)

    def list_feature_groups(self) -> list[int]:
        """List all feature groups stored in the assay."""
        return self._table.list_feature_groups()

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Apply patches to descriptors."""
        self._table.patch_descriptors(*patches)

    def copy_snapshot(self, copy_id: str) -> OnMemoryAssayStorageSnapshot:
        """Create a snapshot copy."""
        res = OnMemoryAssayStorageSnapshot(copy_id, self.status.model_copy(deep=True))
        descriptor_list = self.fetch_descriptors(copy=True)
        annotations = self.fetch_annotations(copy=True)
        fill_values = self.fetch_fill_values(copy=True)
        descriptors_names = list(descriptor_list)
        descriptors = [dict(zip(descriptors_names, values)) for values in zip(*descriptor_list.values())]
        res.add_descriptors(descriptors, annotations)
        res.add_fill_values(fill_values)
        return res
