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
from ..utils.common import create_id

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

        descriptors = dict()
        annotations = list()
        for ft in features:
            for descriptor, value in ft.describe().items():
                descriptor_list = descriptors.setdefault(descriptor, list())
                descriptor_list.append(value)
            annotations.append(ft.annotation)

        annotations = [x.annotation for x in features]
        self._current.add_descriptors(descriptors, annotations)

        self._rois_to_sample_id.update(new_rois)
        self._sample_data[sample.id] = copy

    def add_fill_values(self, *fill_values: FillValue) -> None:
        """Add values to fill missing data matrix entries."""
        d = dict()
        for fill in fill_values:
            sample_d = d.setdefault(fill.sample_id, dict())
            sample_d[fill.feature_group] = fill.value
        self._current.add_fill_values(d)

    def add_feature_groups(self, *feature_groups: FeatureGroup) -> None:
        """Add feature groups to the assay."""
        raise NotImplementedError

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups from the assay."""
        raise NotImplementedError

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

    def fetch_annotations(self, sample_id: str | None = None) -> list[Annotation]:
        """Fetch a copy of the feature annotations.

        :param sample_id: If provided, only fetch annotations from this sample. By default, fetch annotations
            from all samples.
        :raise SampleNotFound: if a sample id that is not in the assay storage is provided.

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
        :raise SampleNotFound: if a sample id that is not in the assay storage is provided.
        :raise InvalidFeatureDescriptor: If an undefined descriptor name for the assay feature type is provided.
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

    def __init__(self, sample: Sample, roi_type: type[RoiType], feature_type: type[FeatureType]) -> None:
        self.roi_type = roi_type
        self.feature_type = feature_type
        self._sample = sample.model_copy()
        latest = OnMemorySampleStorageSnapshot(sample, LATEST, SampleProcessStatus(), roi_type, feature_type)
        self._snapshots = [latest]
        self._current = latest

    def add_features(self, *features: FeatureType) -> None:
        """Add features to the sample storage.

        :param features: the features to be add.
        :raises RepeatedIdError: if a feature with an existing id is provided.
        :raises RoiNotFoundError: if trying to add a feature associated with a ROI not in the storage

        """
        self._check_latest_snapshot()
        self._current.add_features(*features)

    def add_rois(self, *rois: RoiType) -> None:
        """Add ROIs to the sample storage.

        :param rois: the rois to be add
        :raises RepeatedIdError: if a ROI with this id already exists.

        """
        self._check_latest_snapshot()
        self._current.add_rois(*rois)

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

        latest = self._current.copy(LATEST, set_new_ids=True)

        self._snapshots.append(latest)
        self._current = latest

    def delete_features(self, *feature_ids: UUID) -> None:
        """Delete features using their ids.

        Non-existing ids are ignored.

        """
        self._check_latest_snapshot()
        self._current.delete_features(*feature_ids)

    def delete_rois(self, *roi_ids: UUID) -> None:
        """Delete ROIs using their ids.

        Non-existing ids are ignored.

        """
        self._check_latest_snapshot()
        self._current.delete_rois(*roi_ids)

    def get_feature(self, feature_id: UUID) -> FeatureType:
        """Retrieve a feature by id.

        :raises FeatureNotFoundError: if the provided `feature_id` is not in the storage

        """
        return self._current.get_feature(feature_id)

    def get_n_features(self) -> int:
        """Get the total number of features in the storage."""
        return self._current.get_n_features()

    def get_n_rois(self) -> int:
        """Get the total number of ROIs in the storage."""
        return self._current.get_n_rois()

    def get_roi(self, roi_id: UUID) -> RoiType:
        """Retrieve a ROI by id.

        :raises RoiNotFoundError: if the provided `roi_id` is not in the storage

        """
        return self._current.get_roi(roi_id)

    def get_sample(self) -> Sample:
        """Retrieve the storage sample."""
        return self._sample

    def get_snapshot_id(self) -> str:
        """Get the current snapshot id."""
        return self._current.id

    def get_status(self) -> SampleProcessStatus:
        """Get the current process status."""
        return self._current.status

    def has_feature(self, feature_id: UUID) -> bool:
        """Check the existence of a feature using its id."""
        return self._current.has_feature(feature_id)

    def has_roi(self, roi_id: UUID) -> bool:
        """Check the existence of a ROI with the specified id."""
        return self._current.has_roi(roi_id)

    def list_features(self, roi_id: UUID | None = None) -> list[FeatureType]:
        """List stored features.

        :param roi_id: if provided, only features associated with this ROI are listed
        :raises RoiNotFoundError: if the provided `roi_id` is not in the storage

        """
        return self._current.list_features(roi_id)

    def list_rois(self) -> list[RoiType]:
        """List all stored ROIs."""
        return self._current.list_rois()

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
            self._current.id = LATEST

    def set_status(self, status: SampleProcessStatus) -> None:
        """Set the current process status."""
        self._check_latest_snapshot()
        self._current.status = status

    def _check_latest_snapshot(self):
        if self._current.id != LATEST:
            msg = f"Only latest snapshot can be modified. Cannot change the state of snapshot {self._current.id}"
            raise exceptions.SnapshotError(msg)

    @classmethod
    def from_sample_storage(
        cls, sample_storage: SampleStorage[RoiType, FeatureType]
    ) -> OnMemorySampleStorage[RoiType, FeatureType]:
        """Create a new instance using the provided sample storage."""
        sample = sample_storage.get_sample()
        copied = OnMemorySampleStorage(sample, sample_storage.roi_type, sample_storage.feature_type)
        copied._snapshots = list()

        current_snapshot_id = sample_storage.get_snapshot_id()

        for snapshot_id in sample_storage.list_snapshots():
            sample_storage.set_snapshot(snapshot_id)
            snapshot = OnMemorySampleStorageSnapshot.from_sample_storage(sample_storage)
            copied._snapshots.append(snapshot)
        copied.set_snapshot(LATEST)

        sample_storage.set_snapshot(current_snapshot_id)
        return copied


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

        self._descriptors: dict[str, list[float]] = dict()
        self._annotations: list[Annotation] = list()
        self._fill_values: dict[str, dict[int, float]] = dict()
        self._feature_groups: dict[int, FeatureGroup]

        self._sample_to_features: dict[str, set[UUID]] = dict()
        self._group_to_features: dict[int, set[UUID]] = dict()

        # maintains feature index in the snapshot for fast access
        self._feature_to_index: dict[UUID, int] = dict()
        self._current_index = 0

    def add_descriptors(self, descriptors: dict[str, list[float]], annotations: list[Annotation]):
        """Add annotations and descriptors from a sample."""
        for name, values in descriptors.items():
            descriptor_list = self._descriptors.setdefault(name, list())
            descriptor_list.extend(values)
        self._annotations.extend(annotations)

        for k, ann in enumerate(annotations, start=self._current_index):
            sample_set = self._sample_to_features.setdefault(ann.sample_id, set())
            sample_set.add(ann.id)
            if ann.group > -1:
                group_set = self._group_to_features.setdefault(ann.group, set())
                group_set.add(ann.id)

            self._feature_to_index[ann.id] = k

        self._current_index += len(descriptors)

    def add_feature_groups(self, *feature_groups: FeatureGroup) -> None:
        """Add feature groups to the snapshot."""
        for group in feature_groups:
            self._feature_groups[group.group] = group

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups from the snapshot."""
        return [x.model_copy(deep=True) for x in self._feature_groups.values()]

    def has_feature_group(self, group: int) -> bool:
        """Check if a group with the provided id is stored in the assay."""
        return group in self._feature_groups

    def fetch_annotations(self, sample_id: str | None = None, copy: bool = False) -> list[Annotation]:
        """Create a list feature annotations.

        :param sample_id: If provided, only include annotations from this sample
        """
        if sample_id is None:
            result = [x for x in self._annotations]
        else:
            indices = [self._feature_to_index[x] for x in self._sample_to_features[sample_id]]
            result = [self._annotations[x] for x in indices]

        if copy:
            result = [x.model_copy() for x in result]

        return result

    def fetch_descriptors(
        self, descriptors: Iterable[str], sample_id: str | None = None, copy: bool = False
    ) -> dict[str, list[float]]:
        """Fetch descriptors from the snapshot."""
        if sample_id is None:
            result = {x: self._descriptors[x] for x in descriptors}
        else:
            indices = [self._feature_to_index[x] for x in self._sample_to_features[sample_id]]
            result = dict()
            for name in descriptors:
                values = self._descriptors[name]
                result[name] = [values[x] for x in indices]
        if copy:
            result = {k: v.copy() for k, v in result.items()}
        return result

    def fetch_fill_values(self, copy: bool = False) -> dict[str, dict[int, float]]:
        """Fetch snapshot fill values."""
        if copy:
            return {k: v.copy() for k, v in self._fill_values.items()}
        return self._fill_values

    def add_fill_values(self, fill_values: dict[str, dict[int, float]]) -> None:
        """Add missing values to the snapshot."""
        for sample_id, features_fill in fill_values.items():
            for feature_group, value in features_fill.items():
                sample_fill = self._fill_values.setdefault(sample_id, dict())
                sample_fill[feature_group] = value

    def get_sample_id(self, feature_id: UUID) -> str:
        """Retrieve the sample id of a feature."""
        index = self._feature_to_index[feature_id]
        return self._annotations[index].sample_id

    def get_ids_by_group(self, group: int) -> list[UUID]:
        """Retrieve all feature ids associated with a feature group."""
        if group not in self._group_to_features:
            raise exceptions.FeatureGroupNotFound(group)
        return [x for x in self._group_to_features[group]]

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature is in the snapshot."""
        return feature_id in self._feature_to_index

    def patch_annotation(self, *patches: AnnotationPatch) -> None:
        """Apply patches to annotations."""
        for p in patches:
            if not self.has_feature(p.id):
                raise exceptions.FeatureNotFound(p.id)

        for p in patches:
            index = self._feature_to_index[p.id]
            ann = self._annotations[index]
            setattr(ann, p.field, p.value)

    def list_feature_groups(self) -> list[int]:
        """List all feature groups stored in the assay."""
        return list(self._feature_groups)

    def patch_descriptors(self, *patches: DescriptorPatch) -> None:
        """Apply patches to descriptors."""
        for p in patches:
            if not self.has_feature(p.id):
                raise exceptions.FeatureNotFound(p.id)

        for p in patches:
            index = self._feature_to_index[p.id]
            self._descriptors[p.descriptor][index] = p.value

    def copy_snapshot(self, copy_id: str) -> OnMemoryAssayStorageSnapshot:
        """Create a snapshot copy."""
        res = OnMemoryAssayStorageSnapshot(copy_id, self.status.model_copy(deep=True))
        descriptors = self.fetch_descriptors(list(self._descriptors), copy=True)
        annotations = self.fetch_annotations(copy=True)
        fill_values = self.fetch_fill_values(copy=True)
        res.add_descriptors(descriptors, annotations)
        res.add_fill_values(fill_values)
        return res
