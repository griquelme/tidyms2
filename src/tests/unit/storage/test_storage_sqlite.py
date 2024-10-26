import pytest
from sqlalchemy.exc import IntegrityError

from tidyms2.core import exceptions
from tidyms2.core.dataflow import AssayProcessStatus
from tidyms2.core.models import AnnotationPatch, DescriptorPatch
from tidyms2.storage import sqlite
from tidyms2.storage.memory import OnMemorySampleStorage
from tidyms2.storage.sqlite.assay import LATEST_SNAPSHOT
from tidyms2.utils.common import create_id

from .. import helpers
from ..helpers import ConcreteFeature, ConcreteRoi


class AssayStorageFixtures:
    @pytest.fixture(scope="class")
    def assay(self):
        return sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)

    @pytest.fixture(scope="class")
    def data_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("test-on-memory-sample-storage-snapshot")

    @pytest.fixture(scope="class")
    def sample_data1(self, data_path):
        sample = helpers.create_sample(data_path, 1)
        data = OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)
        roi1 = helpers.create_roi(sample)
        roi2 = helpers.create_roi(sample)
        data.add_rois(roi1, roi2)
        ft1 = helpers.create_feature(roi1)
        ft2 = helpers.create_feature(roi2)
        ft1.annotation.group = 0  # type: ignore
        ft2.annotation.group = 0  # type: ignore
        ft3 = helpers.create_feature(roi2)
        data.add_features(ft1, ft2, ft3)
        return data

    @pytest.fixture(scope="class")
    def sample_data2(self, data_path):
        sample = helpers.create_sample(data_path, 2)
        data = OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)
        roi1 = helpers.create_roi(sample)
        roi2 = helpers.create_roi(sample)
        data.add_rois(roi1, roi2)
        ft1 = helpers.create_feature(roi1)
        ft2 = helpers.create_feature(roi2)
        ft1.annotation.group = 0  # type: ignore
        ft2.annotation.group = 0  # type: ignore
        ft3 = helpers.create_feature(roi2)
        data.add_features(ft1, ft2, ft3)
        return data

    @pytest.fixture(scope="class")
    def assay_with_sample(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)
        return assay

    @pytest.fixture(scope="class")
    def assay_with_two_samples(self, assay, sample_data1, sample_data2):
        assay.add_sample_data(sample_data1)
        assay.add_sample_data(sample_data2)
        return assay


class TestSQLiteAssayStorageOnDisk(AssayStorageFixtures):
    def test_create_ok(self, sample_data1, tmp_path):
        host = tmp_path / "assay.db"
        assay = sqlite.SQLiteAssayStorage("assay", host, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)
        assert True


class TestSQLiteAssayStorage(AssayStorageFixtures):
    def test_add_sample_with_repeated_id_raises_error(self, assay_with_sample, sample_data1):
        with pytest.raises(exceptions.RepeatedIdError):
            assay_with_sample.add_sample_data(sample_data1)

    def test_fetch_sample(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1):
        expected = sample_data1.get_sample()
        actual = assay_with_sample.fetch_sample(expected.id)
        assert actual == expected

    def test_fetch_sample_non_existing_sample_raises_error(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_sample.fetch_sample("non-existing-sample-id")

    def test_has_sample_existing_sample_returns_true(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1):
        assert assay_with_sample.has_sample(sample_data1.get_sample().id)

    def test_has_sample_non_existing_sample_returns_false(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        assert not assay_with_sample.has_sample("non-existing-sample-id")

    def test_list_samples(self, assay_with_sample, sample_data1):
        expected = [sample_data1.get_sample()]
        actual = assay_with_sample.list_samples()
        assert actual == expected

    def test_get_n_samples_empty_assay_returns_zero(self, assay: sqlite.SQLiteAssayStorage):
        expected = 0
        actual = assay.get_n_samples()
        assert actual == expected

    def test_get_n_samples(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        expected = 1
        actual = assay_with_sample.get_n_samples()
        assert actual == expected

    def test_fetch_rois_by_sample(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1):
        expected = sample_data1.list_rois()
        actual = assay_with_sample.fetch_rois_by_sample(sample_data1.get_sample().id)
        assert actual
        assert all(x.equal(y) for x, y in zip(expected, actual))

    def test_fetch_rois_by_sample_invalid_sample_raise_error(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_sample.fetch_rois_by_sample("invalid-sample-id")

    def test_fetch_rois_by_ids(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1):
        expected = [sample_data1.list_rois()[1]]
        actual = assay_with_sample.fetch_rois_by_id(*(x.id for x in expected))
        assert actual
        assert all(x.equal(y) for x, y in zip(expected, actual))

    def test_fetch_has_roi_true(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1):
        expected_id = sample_data1.list_rois()[0].id
        assert assay_with_sample.has_roi(expected_id)

    def test_fetch_has_roi_false(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        invalid_id = create_id()
        assert not assay_with_sample.has_roi(invalid_id)

    def test_get_n_rois(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        expected = 2
        actual = assay_with_sample.get_n_rois()
        assert actual == expected

    def test_get_n_rois_empty_assay_returns_zero(self, assay: sqlite.SQLiteAssayStorage):
        expected = 0
        actual = assay.get_n_rois()
        assert actual == expected

    def test_fetch_features_by_group(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        group = 0
        actual = assay_with_sample.fetch_features_by_group(group)
        assert len(actual) == 2
        assert all(ft.annotation.group == group for ft in actual)

    def test_fetch_features_by_id(self, assay_with_sample, sample_data1):
        expected = sample_data1.list_features()
        actual = assay_with_sample.fetch_features_by_id(*(x.id for x in expected))
        assert all(x.equal(y) for x, y in zip(actual, expected))

    def test_fetch_features_by_sample_with_shared_roi_share_the_same_roi_instance(
        self, assay_with_sample, sample_data1
    ):
        sample_id = sample_data1.get_sample().id
        ft1, ft2, ft3 = assay_with_sample.fetch_features_by_sample(sample_id)
        assert ft2.roi == ft3.roi
        assert ft2.roi is ft3.roi
        assert ft1.annotation.sample_id == sample_id
        assert ft2.annotation.sample_id == sample_id
        assert ft3.annotation.sample_id == sample_id

    def test_get_n_features_empty_assay_returns_zero(self, assay: sqlite.SQLiteAssayStorage):
        actual = 0
        expected = assay.get_n_features()
        assert actual == expected

    def test_get_n_features(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        actual = 3
        expected = assay_with_sample.get_n_features()
        assert actual == expected

    def test_has_feature_true(self, assay_with_sample: sqlite.SQLiteAssayStorage, sample_data1: OnMemorySampleStorage):
        ft_id = sample_data1.list_features()[1].id
        assert assay_with_sample.has_feature(ft_id)

    def test_has_feature_false(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        ft_id = create_id()
        assert not assay_with_sample.has_feature(ft_id)

    def test_fetch_sample_data_retrieves_equal_data(self, assay_with_sample, sample_data1):
        expected = sample_data1
        sample_id = expected.get_sample().id
        actual = assay_with_sample.fetch_sample_data(sample_id)

        assert actual.list_snapshots() == expected.list_snapshots()
        for snapshot_id in actual.list_snapshots():
            actual.set_snapshot(snapshot_id)
            expected.set_snapshot(snapshot_id)

            assert actual.list_rois() == expected.list_rois()
            assert actual.list_features() == expected.list_features()
            assert actual.get_status() == expected.get_status()
            assert actual.get_sample() == expected.get_sample()


class TestSQLiteSnapshots(AssayStorageFixtures):
    def test_get_process_state(self, assay: sqlite.SQLiteAssayStorage):
        actual = assay.get_process_status()
        expected = AssayProcessStatus()
        assert actual == expected

    def test_set_process_state(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        expected = AssayProcessStatus(feature_matched=True)
        assert not assay.get_process_status() == expected
        assay.set_process_status(expected)
        assert assay.get_process_status() == expected

    def test_create_snapshot(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.create_snapshot("snap")
        assert ["snap", LATEST_SNAPSHOT] == assay.list_snapshots()

    def test_create_snapshot_with_repeated_id_raises_error(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.create_snapshot("snap")
        with pytest.raises(IntegrityError):
            assay.create_snapshot("snap")

    def test_set_snapshot_ok(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        snapshot_id = "snap1"
        assay.create_snapshot(snapshot_id)
        assay.set_snapshot(snapshot_id)
        assert assay.get_snapshot_id() == snapshot_id

    def test_create_snapshot_from_non_latest_snapshot_raises_error(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.create_snapshot("snap1")
        assay.set_snapshot("snap1")
        with pytest.raises(exceptions.SnapshotError):
            assay.create_snapshot("snap2")

    def test_set_snapshot_and_try_to_modify_state_raises_error(self):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.create_snapshot("snap1")
        assay.set_snapshot("snap1")
        with pytest.raises(exceptions.SnapshotError):
            assay.set_process_status(AssayProcessStatus())

    def test_set_snapshot_and_fetch_annotations(self, assay_with_sample: sqlite.SQLiteAssayStorage):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        annotations_before_set = assay.fetch_annotations()
        assay.create_snapshot("snap1")
        assay.set_snapshot("snap1")
        annotations_after_set = assay.fetch_annotations()
        assert annotations_before_set == annotations_after_set


class TestSQLiteAssayStorageAnnotationApi(AssayStorageFixtures):
    def test_fetch_annotations(self, assay_with_two_samples):
        actual = assay_with_two_samples.fetch_annotations()
        assert len(actual) == assay_with_two_samples.get_n_features()

    def test_fetch_annotations_from_single_sample(self, assay_with_two_samples, sample_data1):
        sample_id = sample_data1.get_sample().id
        actual = assay_with_two_samples.fetch_annotations(sample_id=sample_id)
        assert len(actual) == sample_data1.get_n_features()

    def test_fetch_annotations_from_invalid_sample_raises_error(self, assay_with_two_samples):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_two_samples.fetch_annotations(sample_id="invalid-sample-id")

    def test_patch_annotations_update_data(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)
        annotations_before_patch = assay.fetch_annotations()
        index = 1
        id_ = annotations_before_patch[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        assay.patch_annotations(patch)
        annotations_after_patch = assay.fetch_annotations()
        assert annotations_before_patch != annotations_after_patch
        assert annotations_after_patch[index].group == value

    def test_patch_annotation_and_set_snapshot_previous_snapshot_does_not_apply_patch(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)

        annotations_before_patch = assay.fetch_annotations()
        snapshot_id = "snap"
        assay.create_snapshot(snapshot_id)
        index = 1
        id_ = annotations_before_patch[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        assay.patch_annotations(patch)
        assay.set_snapshot(snapshot_id)
        annotations_from_snapshot = assay.fetch_annotations()
        assert annotations_before_patch == annotations_from_snapshot

    def test_patch_annotation_and_set_snapshot_applies_patch_if_patch_is_from_previous_snapshot(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)

        annotations_before_patch = assay.fetch_annotations()
        snapshot_id = "snap"
        index = 1
        id_ = annotations_before_patch[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        assay.patch_annotations(patch)
        assay.create_snapshot(snapshot_id)

        annotations_after_patch = assay.fetch_annotations()

        assay.set_snapshot(snapshot_id)
        annotations_from_snapshot = assay.fetch_annotations()
        assert annotations_before_patch != annotations_after_patch
        assert annotations_after_patch == annotations_from_snapshot


class TestSQLiteAssayStorageDescriptorApi(AssayStorageFixtures):
    def test_fetch_descriptors(self, assay_with_two_samples):
        actual = assay_with_two_samples.fetch_descriptors()
        assert len(actual["mz"]) == assay_with_two_samples.get_n_features()

    def test_fetch_descriptors_from_single_sample(self, assay_with_two_samples, sample_data1):
        sample_id = sample_data1.get_sample().id
        actual = assay_with_two_samples.fetch_descriptors(sample_id=sample_id)
        assert len(actual["mz"]) == sample_data1.get_n_features()

    def test_fetch_annotations_from_invalid_sample_raises_error(self, assay_with_two_samples):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_two_samples.fetch_descriptors(sample_id="invalid-sample-id")

    def test_patch_descriptors_update_data(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)
        annotations = assay.fetch_annotations()
        descriptors_before_patch = assay.fetch_descriptors()
        index = 1
        id_ = annotations[index].id
        descriptor = "height"
        value = 1000.0
        patch = DescriptorPatch(id=id_, descriptor=descriptor, value=value)
        assay.patch_descriptors(patch)
        descriptors_after_patch = assay.fetch_descriptors()
        assert descriptors_before_patch != descriptors_after_patch
        assert descriptors_after_patch[descriptor][index] == value

    def test_patch_descriptors_and_set_snapshot_previous_snapshot_does_not_apply_patch(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)

        annotations = assay.fetch_annotations()
        descriptors_before_patch = assay.fetch_descriptors()
        snapshot_id = "snap"
        assay.create_snapshot(snapshot_id)
        index = 1
        id_ = annotations[index].id
        descriptor = "height"
        value = 1000.0
        patch = DescriptorPatch(id=id_, descriptor=descriptor, value=value)
        assay.patch_descriptors(patch)
        assay.set_snapshot(snapshot_id)
        descriptors_from_snapshot = assay.fetch_descriptors()
        assert descriptors_before_patch == descriptors_from_snapshot
        assert descriptors_from_snapshot[descriptor][index] != value

    def test_patch_descriptor_and_set_snapshot_applies_patch_if_patch_is_from_previous_snapshot(self, sample_data1):
        assay = sqlite.SQLiteAssayStorage("assay", None, ConcreteRoi, ConcreteFeature)
        assay.add_sample_data(sample_data1)

        annotations = assay.fetch_annotations()
        descriptors_before_patch = assay.fetch_descriptors()
        snapshot_id = "snap"
        index = 1
        id_ = annotations[index].id
        field = "height"
        value = 1000.0
        patch = DescriptorPatch(id=id_, descriptor=field, value=value)
        assay.patch_descriptors(patch)
        assay.create_snapshot(snapshot_id)

        descriptors_after_patch = assay.fetch_descriptors()

        assay.set_snapshot(snapshot_id)
        annotations_from_snapshot = assay.fetch_descriptors()
        assert descriptors_before_patch != descriptors_after_patch
        assert descriptors_after_patch == annotations_from_snapshot
