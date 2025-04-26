"""SQLite based storage classes."""

from typing import Generic, Iterable
from uuid import UUID

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import text

from ...core import exceptions, models
from ...core.dataflow import AssayProcessStatus, SampleProcessStatus
from ...core.models import (
    Annotation,
    AnnotationPatch,
    DescriptorPatch,
    FeatureGroup,
    FeatureType,
    FillValue,
    RoiType,
    Sample,
)
from ...core.storage import SampleStorage
from ..memory import OnMemorySampleStorage
from ..table import FeatureTable
from .models import (
    AnnotationModel,
    AnnotationPatchModel,
    AssaySnapshotStatusModel,
    Base,
    DescriptorModel,
    DescriptorPatchModel,
    FeatureGroupModel,
    FeatureModel,
    FillValueModel,
    RoiModel,
    SampleModel,
    SampleSnapshotModel,
)
from .session import create_session

LATEST_SNAPSHOT = "head"


class SQLiteAssayStorage(Generic[FeatureType, RoiType]):
    """Assay storage class for that persists data using a SQLite backend.

    :param id: an identifier for the storage
    :param host: the DB host string. If not provided an in-memory database is used
    :param roi_type: the ROI class stored in the DB.
    :param feature_type: the feature class stored in the DB.

    """

    def __init__(self, id: str, host: str | None, roi_type: type[RoiType], feature_type: type[FeatureType]):
        self.id = id
        self.roi_type = roi_type
        self.feature_type = feature_type
        self._table: FeatureTable | None = None
        self._current_snapshot_id = LATEST_SNAPSHOT

        if host is None:
            host = "sqlite:///:memory:"
        else:
            host = f"sqlite:///{host}"

        self.engine = create_engine(host)
        self.session_factory = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)

        # create a snapshot if it does not exist
        stmt = select(AssaySnapshotStatusModel).where(AssaySnapshotStatusModel.id == self._current_snapshot_id)
        with create_session(self.session_factory) as session:
            status = session.execute(stmt).scalar()
            if status is None:
                initial_state = AssayProcessStatus()
                self._add_assay_snapshot_meta(session, LATEST_SNAPSHOT, initial_state)

    def add_feature_groups(self, *groups: FeatureGroup) -> None:
        """Add feature groups data to the assay storage."""
        with create_session(self.session_factory) as session:
            session.add_all(FeatureGroupModel.from_feature_group(x) for x in groups)

    def add_fill_values(self, *fill_values: FillValue) -> None:
        """Add values to fill missing entries in the data matrix."""
        table = self._fetch_feature_table()
        d = dict()
        for fill in fill_values:
            sample_d = d.setdefault(fill.sample_id, dict())
            sample_d[fill.feature_group] = fill.value
        table.add_fill_values(d)

        snapshot_id = self.get_snapshot_id()

        with create_session(self.session_factory) as session:
            session.add_all(FillValueModel.from_fill_value(x, snapshot_id) for x in fill_values)

    def add_sample_data(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        """Add sample data to DB."""
        with create_session(self.session_factory) as session:
            sample = data.get_sample()
            if self.has_sample(sample.id):
                raise exceptions.RepeatedIdError(f"Sample with id {sample.id} already stored in the assay.")
            self._add_sample(session, sample)
            self._add_rois(session, data)
            self._add_features(session, data)
            self._add_sample_snapshot_meta(session, data)

    def create_snapshot(self, snapshot_id: str) -> None:
        """Create a new assay data snapshot."""
        self._check_current_snapshot_is_latest_snapshot()
        with create_session(self.session_factory) as session:
            current = self._fetch_current_status(session)
            new_snap = AssaySnapshotStatusModel(id=LATEST_SNAPSHOT, status=self.get_process_status().model_dump_json())
            current.id = snapshot_id
            session.add(new_snap)

            annotation_update = (
                update(AnnotationPatchModel)
                .where(AnnotationPatchModel.snapshot_id == LATEST_SNAPSHOT)
                .values(snapshot_id=snapshot_id)
            )
            session.execute(annotation_update)

            descriptor_update = (
                update(DescriptorPatchModel)
                .where(DescriptorPatchModel.snapshot_id == LATEST_SNAPSHOT)
                .values(snapshot_id=snapshot_id)
            )
            session.execute(descriptor_update)

            fill_value_update = (
                update(FillValueModel)
                .where(FillValueModel.snapshot_id == LATEST_SNAPSHOT)
                .values(snapshot_id=snapshot_id)
            )
            session.execute(fill_value_update)

    def fetch_annotations(self, sample_id: str | None = None) -> list[Annotation]:
        """Fetch the feature annotations."""
        table = self._fetch_feature_table()
        return table.fetch_annotations(sample_id=sample_id, copy=True)

    def fetch_descriptors(self, sample_id: str | None = None, descriptors: Iterable[str] | None = None) -> dict:
        """Fetch the feature descriptors."""
        table = self._fetch_feature_table()
        all_descriptors = self.feature_type.descriptor_names()
        if descriptors is None:
            descriptors = list(all_descriptors)
        for x in descriptors:
            if x not in all_descriptors:
                raise exceptions.InvalidFeatureDescriptor(x)
        return table.fetch_descriptors(descriptors=descriptors, sample_id=sample_id, copy=True)

    def fetch_features_by_group(self, group: int) -> list[FeatureType]:
        """Fetch features using the feature group id."""
        stmt = (
            select(FeatureModel, AnnotationModel)
            .join(AnnotationModel, FeatureModel.id == AnnotationModel.id)
            .where(AnnotationModel.group == group)
            .where(FeatureModel.snapshot_id == LATEST_SNAPSHOT)
        )
        with create_session(self.session_factory) as session:
            results = [row._tuple() for row in session.execute(stmt).all()]
            features = self._fetch_features_from_model(session, results)

        return features

    def fetch_features_by_id(self, *feature_ids: UUID) -> list[FeatureType]:
        """Fetch features using their ids."""
        stmt = (
            select(FeatureModel, AnnotationModel)
            .join(AnnotationModel, FeatureModel.id == AnnotationModel.id)
            .where(FeatureModel.id.in_(feature_ids))
            .where(FeatureModel.snapshot_id == LATEST_SNAPSHOT)
        )
        with create_session(self.session_factory) as session:
            results = [row._tuple() for row in session.execute(stmt).all()]
            features = self._fetch_features_from_model(session, results)

        return features

    def fetch_features_by_sample(self, sample_id: str) -> list[FeatureType]:
        """Fetch features using the sample id."""
        stmt = (
            select(FeatureModel, AnnotationModel)
            .join(AnnotationModel, FeatureModel.id == AnnotationModel.id)
            .where(AnnotationModel.sample_id == sample_id)
            .where(FeatureModel.snapshot_id == LATEST_SNAPSHOT)
        )
        with create_session(self.session_factory) as session:
            results = [row._tuple() for row in session.execute(stmt).all()]
            features = self._fetch_features_from_model(session, results)

        return features

    def fetch_feature_groups(self) -> list[FeatureGroup]:
        """Fetch feature groups stored in the assay."""
        with create_session(self.session_factory) as session:
            stmt = select(FeatureGroupModel)
            feature_groups = [x.to_feature_group() for x in session.execute(stmt).scalars()]
        return feature_groups

    def fetch_fill_values(self) -> dict[str, dict[int, float]]:
        """Fetch fill values for data matrix."""
        table = self._fetch_feature_table()
        return table.fetch_fill_values(copy=True)

    def fetch_rois_by_id(self, *roi_ids: UUID) -> list[RoiType]:
        """Fetch ROIs using their ids.

        :param roi_ids: a list of ROI ids to fetch

        """
        with create_session(self.session_factory) as session:
            rois = self._fetch_rois_by_id(session, LATEST_SNAPSHOT, *roi_ids)
        return rois

    def fetch_rois_by_sample(self, sample_id: str) -> list[RoiType]:
        """Fetch all ROIs from a sample."""
        with create_session(self.session_factory) as session:
            sample = self._fetch_sample(session, sample_id)
            stmt = (
                select(RoiModel)
                .where((RoiModel.sample_id == sample_id))
                .where(RoiModel.snapshot_id == LATEST_SNAPSHOT)
            )
            models = session.execute(stmt).scalars()
            rois = [self.roi_type.from_str(x.data, sample) for x in models]
        return rois

    def get_feature_type(self) -> type[FeatureType]:
        """Retrieve the Feature class used."""
        return self.feature_type

    def get_roi_type(self) -> type[RoiType]:
        """Retrieve the ROI class used."""
        return self.roi_type

    def get_n_rois(self) -> int:
        """Retrieve the number of ROIs in the assay."""
        with create_session(self.session_factory) as session:
            stmt = text("SELECT COUNT(1) FROM rois;")
            n_rois = session.execute(stmt).scalar_one_or_none()
        n_rois = 0 if n_rois is None else n_rois
        return n_rois

    def get_n_features(self) -> int:
        """Retrieve the number of features in the assay."""
        with create_session(self.session_factory) as session:
            stmt = text("SELECT COUNT(1) FROM features;")
            n_fts = session.execute(stmt).scalar_one_or_none()
        n_fts = 0 if n_fts is None else n_fts
        return n_fts

    def get_n_samples(self) -> int:
        """Retrieve the number of samples in the assay."""
        with create_session(self.session_factory) as session:
            stmt = text("SELECT COUNT(1) FROM samples;")
            n_samples = session.execute(stmt).scalar_one_or_none()
        n_samples = 0 if n_samples is None else n_samples
        return n_samples

    def get_process_status(self) -> AssayProcessStatus:
        """Get the current process status."""
        with create_session(self.session_factory) as session:
            status = self._fetch_current_status(session).to_status()
        return status

    def get_snapshot_id(self) -> str:
        """Retrieve the current snapshot id."""
        return self._current_snapshot_id

    def fetch_sample(self, sample_id: str) -> Sample:
        """Retrieve a sample from the assay.

        :param sample_id: the id of the sample to retrieve
        :raises SampleNotFound: if the provided id is not found in the DB

        """
        with create_session(self.session_factory) as session:
            sample = self._fetch_sample(session, sample_id)
        return sample

    def fetch_sample_data(self, sample_id: str) -> OnMemorySampleStorage[RoiType, FeatureType]:
        """Fetch Samples from the assay using their ids."""
        with create_session(self.session_factory) as session:
            sample = self._fetch_sample(session, sample_id)
            all_rois = self._fetch_all_rois(session, sample)
            all_features = self._fetch_all_features(session, sample, all_rois)
            snapshot_ids = self._fetch_sample_snapshot_ids(session, sample.id)
            states = self._fetch_sample_states(session, sample.id)
            data = OnMemorySampleStorage.from_dict(
                sample, all_rois, all_features, snapshot_ids, states, self.roi_type, self.feature_type
            )

        return data

    def has_feature(self, feature_id: UUID) -> bool:
        """Check if a feature with the provided id is stored in the DB.

        :param feature_id: the id of the ROI to check

        """
        with create_session(self.session_factory) as session:
            stmt = select(FeatureModel).where(FeatureModel.id == feature_id)
            check = session.execute(stmt).scalars().first() is not None
        return check

    def has_feature_group(self, feature_group: int) -> bool:
        """Check if a group with the provided id is in the assay."""
        raise NotImplementedError

    def has_roi(self, roi_id: UUID) -> bool:
        """Check if a ROI with the provided id is stored in the DB.

        :param roi_id: the id of the ROI to check

        """
        with create_session(self.session_factory) as session:
            stmt = select(RoiModel).where(RoiModel.id == roi_id)
            check = session.execute(stmt).scalars().first() is not None
        return check

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample with the provided id is stored in the DB.

        :param sample_id: the id of the sample to check

        """
        with create_session(self.session_factory) as session:
            stmt = select(SampleModel).where(SampleModel.id == sample_id)
            check = session.execute(stmt).scalars().first() is not None
        return check

    def list_samples(self) -> list[Sample]:
        """List samples in the assay."""
        with create_session(self.session_factory) as session:
            samples = self._fetch_all_samples(session)
        return samples

    def list_feature_groups(self) -> list[int]:
        """List all group ids in the assay."""
        raise NotImplementedError

    def list_snapshots(self) -> list[str]:
        """Retrieve the list of all snapshots."""
        with create_session(self.session_factory) as session:
            snaps = self._list_snapshots(session)
        return snaps

    def _list_snapshots(self, session: Session) -> list[str]:
        """Retrieve the list of all snapshots."""
        stmt = select(AssaySnapshotStatusModel.id).order_by(AssaySnapshotStatusModel.order)
        return [x for x in session.execute(stmt).scalars()]

    def patch_annotations(self, *patches: models.AnnotationPatch) -> None:
        """Update feature annotation values."""
        self._check_current_snapshot_is_latest_snapshot()
        table = self._fetch_feature_table()
        table.patch_annotation(*patches)
        with create_session(self.session_factory) as session:
            self._add_annotation_patches(session, *patches)

    def patch_descriptors(self, *patches: models.DescriptorPatch) -> None:
        """Update feature descriptors values."""
        self._check_current_snapshot_is_latest_snapshot()
        table = self._fetch_feature_table()
        table.patch_descriptors(*patches)
        with create_session(self.session_factory) as session:
            self._add_descriptor_patches(session, *patches)

    def set_process_status(self, status: AssayProcessStatus) -> None:
        """Set the new process status."""
        self._check_current_snapshot_is_latest_snapshot()
        with create_session(self.session_factory) as session:
            stmt = select(AssaySnapshotStatusModel).where(AssaySnapshotStatusModel.id == self._current_snapshot_id)
            current = session.execute(stmt).scalar_one()
            current.status = status.model_dump_json()

    def set_snapshot(self, snapshot_id: str | None = None, reset: bool = False) -> None:
        """Set assay storage data to specified snapshot.

        If ``None``, fetch data from the latest snapshot.

        """
        if snapshot_id is None:
            snapshot_id = LATEST_SNAPSHOT

        if snapshot_id == self._current_snapshot_id and not reset:
            return

        stmt = select(AssaySnapshotStatusModel).where(AssaySnapshotStatusModel.id == snapshot_id)
        with create_session(self.session_factory) as session:
            model = session.execute(stmt).scalar()
            if model is None:
                raise exceptions.SnapshotNotFound(snapshot_id)
            self._current_snapshot_id = snapshot_id
        self._table = None  # remove cached feature descriptor and annotation data

    def _add_sample(self, session: Session, sample: Sample) -> None:
        """Add sample to DB."""
        model = SampleModel.from_sample(sample)
        session.add(model)

    def _add_features(self, session: Session, data: SampleStorage[RoiType, FeatureType]) -> None:
        feature_models = list()
        ann_models = list()
        descriptor_models = list()
        for snapshot_id in data.list_snapshots():
            data.set_snapshot(snapshot_id)
            snapshot_features = data.list_features()
            feature_models.extend([FeatureModel.from_feature(x, snapshot_id) for x in snapshot_features])
            ann_models.extend([AnnotationModel.from_feature(x, snapshot_id) for x in snapshot_features])
            descriptor_models.extend([DescriptorModel.from_feature(x, snapshot_id) for x in snapshot_features])
        data.set_snapshot()
        session.add_all(descriptor_models)
        session.add_all(feature_models)
        session.add_all(ann_models)

    def _add_rois(self, session: Session, data: SampleStorage[RoiType, FeatureType]) -> None:
        roi_models = list()
        for snapshot_id in data.list_snapshots():
            data.set_snapshot(snapshot_id)
            roi_models.extend([RoiModel.from_roi(x, snapshot_id) for x in data.list_rois()])
        data.set_snapshot()
        session.add_all(roi_models)

    def _add_sample_snapshot_meta(self, session: Session, data: SampleStorage[RoiType, FeatureType]) -> None:
        id_ = data.get_sample().id
        snapshots = list()
        current_snapshot_id = data.get_snapshot_id()
        for k, snapshot_id in enumerate(data.list_snapshots()):
            data.set_snapshot(snapshot_id)
            state = data.get_status()
            snap = SampleSnapshotModel(sample_id=id_, order=k, snapshot_id=snapshot_id, status=state.model_dump_json())
            snapshots.append(snap)
        data.set_snapshot(current_snapshot_id)
        session.add_all(snapshots)

    def _add_annotation_patches(self, session: Session, *patches: AnnotationPatch) -> None:
        session.add_all(AnnotationPatchModel.from_patch(x, snapshot=LATEST_SNAPSHOT) for x in patches)

    def _add_descriptor_patches(self, session: Session, *patches: DescriptorPatch) -> None:
        session.add_all(DescriptorPatchModel.from_patch(x, snapshot=LATEST_SNAPSHOT) for x in patches)

    def _check_current_snapshot_is_latest_snapshot(self):
        id_ = self._current_snapshot_id
        if id_ != LATEST_SNAPSHOT:
            msg = f"Only latest snapshot can be modified. Cannot change the state of snapshot {id_}"
            raise exceptions.SnapshotError(msg)

    def _fetch_sample_snapshot_ids(self, session: Session, sample_id: str) -> list[str]:
        stmt = (
            select(SampleSnapshotModel.snapshot_id)
            .where(SampleSnapshotModel.sample_id == sample_id)
            .order_by(SampleSnapshotModel.order)
        )
        return list(session.execute(stmt).scalars().all())

    def _fetch_sample_states(self, session: Session, sample_id: str) -> dict[str, SampleProcessStatus]:
        stmt = select(SampleSnapshotModel).where(SampleSnapshotModel.sample_id == sample_id)
        results = session.execute(stmt).scalars().all()
        return {x.snapshot_id: SampleProcessStatus.model_validate_json(x.status) for x in results}

    def _fetch_features_from_model(self, session: Session, results: list[tuple[FeatureModel, AnnotationModel]]):
        roi_ids = {x[0].roi_id for x in results}
        id_to_roi = {x.id: x for x in self._fetch_rois_by_id(session, LATEST_SNAPSHOT, *roi_ids)}
        features = list()
        for ft_model, ann_model in results:
            roi = id_to_roi[ann_model.roi_id]
            ft = self.feature_type.from_str(ft_model.data, roi, ann_model.to_dict())  # type: ignore
            features.append(ft)
        return features

    def _fetch_all_samples(self, session: Session) -> list[Sample]:
        stmt = select(SampleModel)
        return [x.to_pydantic_model() for x in session.execute(stmt).scalars()]

    def _fetch_sample(self, session: Session, sample_id: str) -> Sample:
        stmt = select(SampleModel).where(SampleModel.id == sample_id)
        model = session.execute(stmt).scalars().first()
        if model is None:
            raise exceptions.SampleNotFound(sample_id)
        return model.to_pydantic_model()

    def _fetch_rois_by_id(self, session: Session, snapshot_id: str, *roi_ids: UUID) -> list[RoiType]:
        stmt = select(RoiModel).where(RoiModel.id.in_(roi_ids)).where(RoiModel.snapshot_id == snapshot_id)
        sample_id_to_sample = {x.id: x for x in self._fetch_all_samples(session)}
        roi_models = session.execute(stmt).scalars()
        rois = [self.roi_type.from_str(x.data, sample_id_to_sample[x.sample_id]) for x in roi_models]
        return rois

    def _fetch_all_rois(self, session: Session, sample: Sample) -> dict[str, list[RoiType]]:
        """Fetch ROIs from all snapshots."""
        stmt = select(RoiModel).where(RoiModel.sample_id == sample.id)
        snapshot_to_rois: dict[str, list[RoiType]] = dict()
        for roi_model in session.execute(stmt).scalars():
            rois = snapshot_to_rois.setdefault(roi_model.snapshot_id, list())
            rois.append(self.roi_type.from_str(roi_model.data, sample))
        return snapshot_to_rois

    def _fetch_all_features(self, session: Session, sample: Sample, snapshot_to_roi: dict[str, list[RoiType]]):
        """Fetch features from all snapshots."""
        stmt = (
            select(FeatureModel, AnnotationModel)
            .join(AnnotationModel, FeatureModel.id == AnnotationModel.id)
            .where(AnnotationModel.sample_id == sample.id)
        )
        snapshot_to_features: dict[str, list[FeatureType]] = dict()

        roi_id_to_roi = dict()
        for roi_list in snapshot_to_roi.values():
            for roi in roi_list:
                roi_id_to_roi[roi.id] = roi

        for row in session.execute(stmt).all():
            ft_model, ann_model = row._tuple()
            features = snapshot_to_features.setdefault(ft_model.snapshot_id, list())

            annotation = Annotation(**ann_model.to_dict())
            roi = roi_id_to_roi[annotation.roi_id]
            features.append(self.feature_type.from_str(ft_model.data, roi, annotation))

        return snapshot_to_features

    def _fetch_feature_table(self) -> FeatureTable:
        if self._table is not None:
            return self._table

        with create_session(self.session_factory) as session:
            descriptors = self._fetch_descriptors(session, LATEST_SNAPSHOT)
            annotations = self._fetch_annotations(session, LATEST_SNAPSHOT)
            ann_patches = self._fetch_annotations_patches(session)
            descriptor_patches = self._fetch_descriptor_patches(session)
            fill_values = self._fetch_fill_values(session)
        self._table = FeatureTable()
        self._table.add_descriptors(descriptors, annotations)
        self._table.add_fill_values(fill_values)
        for patches in ann_patches:
            self._table.patch_annotation(*patches)
        for patches in descriptor_patches:
            self._table.patch_descriptors(*patches)
        return self._table

    def _fetch_descriptors(self, session: Session, snapshot_id: str) -> list[dict[str, float]]:
        stmt = select(DescriptorModel).where(DescriptorModel.snapshot_id == snapshot_id)
        return [x.create_descriptor_dict() for x in session.execute(stmt).scalars()]

    def _fetch_annotations(self, session: Session, snapshot_id: str) -> list[Annotation]:
        stmt = select(AnnotationModel).where(AnnotationModel.snapshot_id == snapshot_id)
        return [x.to_annotation() for x in session.execute(stmt).scalars()]

    def _fetch_annotations_patches(self, session: Session) -> list[list[AnnotationPatch]]:
        current_snapshot = self.get_snapshot_id()
        all_snapshots = self._list_snapshots(session)
        current_index = all_snapshots.index(current_snapshot)
        fetch_snapshots = all_snapshots[: current_index + 1]

        stmt = select(AnnotationPatchModel).where(AnnotationPatchModel.snapshot_id.in_(fetch_snapshots))

        patches = dict()
        for p in session.execute(stmt).scalars():
            snapshot_patches = patches.setdefault(p.snapshot_id, list())
            snapshot_patches.append(AnnotationPatch(id=p.feature_id, field=p.field, value=p.value))

        return [patches.get(x, list()) for x in fetch_snapshots]

    def _fetch_descriptor_patches(self, session: Session) -> list[list[DescriptorPatch]]:
        current_snapshot = self.get_snapshot_id()
        all_snapshots = self._list_snapshots(session)
        current_index = all_snapshots.index(current_snapshot)
        fetch_snapshots = all_snapshots[: current_index + 1]

        stmt = select(DescriptorPatchModel).where(DescriptorPatchModel.snapshot_id.in_(fetch_snapshots))

        patches = dict()
        for p in session.execute(stmt).scalars():
            snapshot_patches = patches.setdefault(p.snapshot_id, list())
            snapshot_patches.append(DescriptorPatch(id=p.feature_id, descriptor=p.descriptor, value=p.value))

        return [patches.get(x, list()) for x in fetch_snapshots]

    def _fetch_fill_values(self, session: Session) -> dict[str, dict[int, float]]:
        current_snapshot = self.get_snapshot_id()
        all_snapshots = self._list_snapshots(session)
        current_index = all_snapshots.index(current_snapshot)
        fetch_snapshots = all_snapshots[: current_index + 1]

        stmt = select(FillValueModel).where(FillValueModel.snapshot_id.in_(fetch_snapshots))

        fill_values = dict()
        for fill in session.execute(stmt).scalars():
            sample_fills = fill_values.setdefault(fill.sample_id, dict())
            sample_fills.setdefault(fill.feature_group, 0.0)
            sample_fills[fill.feature_group] += fill.value
        return fill_values

    def _fetch_current_status(self, session: Session) -> AssaySnapshotStatusModel:
        stmt = select(AssaySnapshotStatusModel).where(AssaySnapshotStatusModel.id == self._current_snapshot_id)
        return session.execute(stmt).scalar_one()

    def _add_assay_snapshot_meta(self, session: Session, snapshot_id: str, state: AssayProcessStatus) -> None:
        model = AssaySnapshotStatusModel(id=snapshot_id, status=state.model_dump_json())
        session.add(model)
