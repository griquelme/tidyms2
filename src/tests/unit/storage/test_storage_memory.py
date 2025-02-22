import pytest

from tidyms2.core import exceptions
from tidyms2.core.dataflow import SampleProcessStatus
from tidyms2.core.models import AnnotationPatch, DescriptorPatch, FillValue
from tidyms2.core.utils.common import create_id
from tidyms2.storage import memory

from .. import helpers


class TestOnMemorySampleStorage:
    @pytest.fixture(scope="class")
    def data_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("test-on-memory-sample-storage-snapshot")

    @pytest.fixture(scope="class")
    def sample(self, data_path):
        return helpers.create_sample(data_path, 1)

    @pytest.fixture(scope="class")
    def roi1(self, sample):
        return helpers.create_roi(sample)

    @pytest.fixture(scope="class")
    def roi2(self, sample):
        return helpers.create_roi(sample)

    @pytest.fixture(scope="class")
    def ft11(self, roi1):
        return helpers.create_feature(roi1)

    @pytest.fixture(scope="class")
    def ft21(self, roi2):
        return helpers.create_feature(roi2)

    @pytest.fixture(scope="class")
    def ft22(self, roi2):
        return helpers.create_feature(roi2)

    def test_create(self, sample):
        memory.OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)

    @pytest.fixture
    def storage(self, sample):
        return memory.OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)

    def test_get_roi_invalid_id_raises_error(self, storage):
        with pytest.raises(exceptions.RoiNotFound):
            invalid_id = create_id()
            storage.get_roi(invalid_id)

    def test_has_roi_invalid_roi_id_raises_error(self, storage, roi1):
        invalid_id = create_id()
        assert not storage.has_roi(invalid_id)

    def test_has_roi_ok(self, storage, roi1):
        storage.add_rois(roi1)
        assert storage.has_roi(roi1.id)

    def test_get_roi(self, storage, roi1):
        expected = roi1
        storage.add_rois(expected)
        actual = storage.get_roi(expected.id)
        assert actual == expected

    def test_add_multiple_roi(self, storage, roi1, roi2):
        storage.add_rois(roi1, roi2)
        assert storage.has_roi(roi1.id)
        assert storage.has_roi(roi2.id)

    def test_add_rois_with_repeated_ids_does_not_add_any_roi(self, storage, roi1, roi2):
        storage.add_rois(roi1)
        with pytest.raises(exceptions.RepeatedIdError):
            storage.add_rois(roi2, roi1)
            assert storage.has_roi(roi1.id)
            assert not storage.has_roi(roi2.id)

    def test_add_roi_with_features_also_does_not_add_features(self, storage, roi1, ft11):
        storage.add_rois(roi1)
        assert storage.has_roi(roi1.id)
        assert not storage.has_feature(ft11.id)

    def test_list_rois_no_roi_returns_empty_list(self, storage):
        result = storage.list_rois()
        assert not result

    def test_list_rois_ok(self, storage, roi1, roi2):
        storage.add_rois(roi1, roi2)
        result = storage.list_rois()
        assert roi1 in result
        assert roi2 in result
        assert len(result) == 2

    def test_delete_single_roi(self, storage, roi1):
        storage.add_rois(roi1)
        assert storage.has_roi(roi1.id)

        storage.delete_rois(roi1.id)
        assert not storage.has_roi(roi1.id)

    def test_delete_multiple_rois(self, storage, roi1, roi2):
        storage.add_rois(roi1)
        storage.add_rois(roi2)

        storage.delete_rois(roi1.id, roi2.id)
        assert not storage.has_roi(roi1.id)
        assert not storage.has_roi(roi2.id)

    def test_delete_non_existing_roi_id_is_ignored(self, storage, roi1, roi2):
        storage.add_rois(roi1)

        storage.delete_rois(roi2.id)

        assert storage.has_roi(roi1.id)
        assert not storage.has_roi(roi2.id)

    def test_delete_roi_with_features_also_delete_features(self, storage, roi1, ft11):
        storage.add_rois(roi1)
        storage.add_features(ft11)

        assert storage.has_roi(roi1.id)
        assert storage.has_feature(ft11.id)

        storage.delete_rois(roi1.id)

        assert not storage.has_roi(roi1.id)
        assert not storage.has_feature(ft11.id)

    def test_add_single_feature(self, storage, roi1):
        storage.add_rois(roi1)

        # check that id is not on storage before creating
        feature_id = create_id()
        assert not storage.has_feature(feature_id)

        # check that feature is not added on creation
        feature = helpers.ConcreteFeature(id=feature_id, roi=roi1, data_mz=1)
        assert not storage.has_feature(feature_id)

        # check ok after add
        storage.add_features(feature)
        assert storage.has_feature(feature_id)

    def test_add_feature_raise_error_if_parent_roi_is_not_stored(self, storage, ft11):
        with pytest.raises(exceptions.RoiNotFound):
            storage.add_features(ft11)

    def test_add_multiple_features(self, storage, roi1):
        storage.add_rois(roi1)

        ft1 = helpers.ConcreteFeature(id=create_id(), roi=roi1, data_mz=1)
        ft2 = helpers.ConcreteFeature(id=create_id(), roi=roi1, data_mz=1)

        assert not storage.has_feature(ft1.id)
        assert not storage.has_feature(ft2.id)

        storage.add_features(ft1, ft2)

        assert storage.has_feature(ft1.id)
        assert storage.has_feature(ft2.id)

    def test_add_feature_with_repeated_id_raises_error(self, storage, roi1, ft11):
        storage.add_rois(roi1)
        storage.add_features(ft11)
        with pytest.raises(exceptions.RepeatedIdError):
            storage.add_features(ft11)

    def test_add_feature_with_parent_roi_not_stored_raises_error(self, storage, roi1, ft11):
        with pytest.raises(exceptions.RoiNotFound):
            storage.add_features(ft11)

    def test_get_feature(self, storage, roi1, ft11):
        storage.add_rois(roi1)
        storage.add_features(ft11)

        expected = ft11
        actual = storage.get_feature(expected.id)

        assert actual == expected

    def test_get_feature_non_existing_id_raises_error(self, storage):
        with pytest.raises(exceptions.FeatureNotFound):
            id_ = create_id()
            storage.get_feature(id_)

    def test_features_no_feature_return_empty_list(self, storage):
        assert not storage.list_features()

    def test_features_all_rois(self, storage, roi1, roi2, ft11, ft21):
        storage.add_rois(roi1, roi2)
        storage.add_features(ft11, ft21)
        features = storage.list_features()
        assert ft11 in features
        assert ft21 in features

    def test_features_single_roi(self, storage, roi1, roi2, ft11, ft21):
        storage.add_rois(roi1, roi2)
        storage.add_features(ft11, ft21)
        features = storage.list_features(roi_id=roi1.id)
        assert ft11 in features
        assert ft21 not in features

    def test_features_non_existing_roi_id_raises_error(self, storage, roi1):
        with pytest.raises(exceptions.RoiNotFound):
            storage.list_features(roi_id=roi1.id)

    def test_delete_single_feature(self, storage, roi1, ft11):
        storage.add_rois(roi1)
        storage.add_features(ft11)

        storage.delete_features(ft11.id)

        assert not storage.has_feature(ft11.id)

    def test_delete_multiple_features(self, storage, roi1, roi2, ft11, ft21, ft22):
        storage.add_rois(roi1, roi2)
        storage.add_features(ft11, ft21, ft22)

        storage.delete_features(ft11.id, ft21.id)

        assert not storage.has_feature(ft11.id)
        assert not storage.has_feature(ft21.id)
        assert storage.has_feature(ft22.id)

    def test_delete_features_non_existing_ids_are_ignored(self, storage, roi1, roi2, ft11, ft21):
        storage.add_rois(roi1, roi2)
        storage.add_features(ft11, ft21)

        non_existing_id = create_id()

        storage.delete_features(non_existing_id)

        assert storage.has_feature(ft11.id)
        assert storage.has_feature(ft21.id)

    @pytest.fixture
    def storage_with_snapshot(self, sample, roi1, ft11):
        storage = memory.OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)
        storage.add_rois(roi1)
        storage.add_features(ft11)
        storage.create_snapshot("snapshot")
        return storage

    def test_create_snapshot_with_reserved_id_raises_error(self, storage):
        with pytest.raises(ValueError):
            storage.create_snapshot(memory.LATEST)

    def test_create_snapshot_sets_current_snapshot_as_latest(self, storage_with_snapshot):
        storage_with_snapshot.create_snapshot("new-snapshot")
        assert storage_with_snapshot.get_snapshot_id() == memory.LATEST

    def test_list_snapshots_no_snapshots_on_creation(self, storage):
        assert storage.list_snapshots() == []

    def test_list_snapshots_on_storage_with_data_but_no_snapshots_has_only_latest_snapshot_id(
        self, storage, roi1, roi2
    ):
        storage.add_rois(roi1, roi2)
        assert storage.list_snapshots() == [memory.LATEST]

    def test_list_snapshots(self, storage_with_snapshot):
        expected = ["snapshot", memory.LATEST]
        actual = storage_with_snapshot.list_snapshots()
        assert actual == expected

    def test_create_snapshot_with_existing_id_raises_error(self, storage_with_snapshot):
        with pytest.raises(exceptions.RepeatedIdError):
            storage_with_snapshot.create_snapshot("snapshot")

    def test_set_snapshot_non_existing_id_raises_error(self, storage_with_snapshot):
        with pytest.raises(exceptions.SnapshotNotFound):
            storage_with_snapshot.set_snapshot("invalid-snapshot-id")

    def test_add_roi_and_set_to_snapshot_rollback_changes(self, storage_with_snapshot, roi2):
        storage_with_snapshot.add_rois(roi2)

        assert storage_with_snapshot.has_roi(roi2.id)

        storage_with_snapshot.set_snapshot("snapshot")

        assert not storage_with_snapshot.has_roi(roi2.id)

    def test_add_feature_and_set_to_snapshot_rollback_changes(self, storage_with_snapshot, roi2, ft21):
        storage_with_snapshot.add_rois(roi2)
        storage_with_snapshot.add_features(ft21)

        assert storage_with_snapshot.has_feature(ft21.id)

        storage_with_snapshot.set_snapshot("snapshot")

        assert not storage_with_snapshot.has_feature(ft21.id)

    def test_try_to_add_roi_to_non_latest_snapshot_raises_error(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        sample = storage_with_snapshot.get_sample()
        dummy_roi = helpers.create_roi(sample)
        with pytest.raises(exceptions.SnapshotError):
            storage_with_snapshot.add_rois(dummy_roi)

    def test_try_to_add_feature_to_non_latest_snapshot_raises_error(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        sample = storage_with_snapshot.get_sample()
        dummy_roi = helpers.create_roi(sample)
        dummy_ft = helpers.create_feature(dummy_roi)
        with pytest.raises(exceptions.SnapshotError):
            storage_with_snapshot.add_features(dummy_ft)

    def test_try_to_set_non_latest_snapshot_status_raises_error(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        new_status = SampleProcessStatus(feature_extracted=True)
        with pytest.raises(exceptions.SnapshotError):
            storage_with_snapshot.set_status(new_status)

    def test_set_status(self, storage):
        expected = SampleProcessStatus(isotopologue_annotated=True)
        storage.set_status(expected)
        actual = storage.get_status()
        assert actual == expected

    def test_set_snapshot_with_reset(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot", reset=True)

        assert "snapshot" not in storage_with_snapshot.list_snapshots()

        with pytest.raises(exceptions.SnapshotNotFound):
            storage_with_snapshot.set_snapshot("snapshot")

    def test_using_snapshot_changes_roi_id(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        roi_snapshot = storage_with_snapshot.list_rois()[0]
        storage_with_snapshot.set_snapshot()
        roi_latest = storage_with_snapshot.list_rois()[0]

        assert roi_snapshot.id != roi_latest.id
        assert roi_latest.equal(roi_snapshot)

    def test_using_snapshot_changes_feature_id(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        feature_snapshot = storage_with_snapshot.list_features()[0]
        storage_with_snapshot.set_snapshot()
        feature_latest = storage_with_snapshot.list_features()[0]

        assert feature_snapshot.id != feature_latest.id
        assert feature_snapshot.roi.id != feature_latest.roi.id
        assert feature_latest.roi.equal(feature_snapshot.roi)
        assert feature_latest.equal(feature_snapshot)

    def test_using_snapshot_modify_rois_does_not_modify_rois_in_snapshots(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        roi_snapshot = storage_with_snapshot.list_rois()[0]
        storage_with_snapshot.set_snapshot()
        roi_latest = storage_with_snapshot.list_rois()[0]

        assert roi_latest.equal(roi_snapshot)
        roi_latest.data.append(10.0)
        assert not roi_latest.equal(roi_snapshot)

    def test_using_snapshot_modify_features_does_not_modify_features_in_snapshots(self, storage_with_snapshot):
        storage_with_snapshot.set_snapshot("snapshot")
        feature_snapshot = storage_with_snapshot.list_features()[0]
        storage_with_snapshot.set_snapshot()
        feature_latest = storage_with_snapshot.list_features()[0]

        assert feature_snapshot.id != feature_latest.id
        assert feature_snapshot.roi.id != feature_latest.roi.id
        assert feature_latest.roi.equal(feature_snapshot.roi)
        assert feature_latest.equal(feature_snapshot)
        feature_latest.data_mz = 1000
        assert not feature_latest.equal(feature_snapshot)

    def test_from_sample_creates_same_sample_data(self, storage_with_snapshot):
        copy = memory.OnMemorySampleStorage.from_sample_storage(storage_with_snapshot)
        assert copy.list_rois() == storage_with_snapshot.list_rois()


class AssayStorageFixtures:
    @pytest.fixture
    def assay(self):
        return memory.OnMemoryAssayStorage("test-assay", helpers.ConcreteRoi, helpers.ConcreteFeature)

    @pytest.fixture(scope="class")
    def data_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("test-on-memory-sample-storage-snapshot")

    @pytest.fixture(scope="class")
    def sample_data1(self, data_path):
        sample = helpers.create_sample(data_path, 1)
        data = memory.OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)
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
        data = memory.OnMemorySampleStorage(sample, helpers.ConcreteRoi, helpers.ConcreteFeature)
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

    @pytest.fixture
    def assay_with_sample_data(self, assay, sample_data1):
        assay.add_sample_data(sample_data1)
        return assay

    @pytest.fixture
    def assay_with_two_samples(self, assay, sample_data1, sample_data2):
        assay.add_sample_data(sample_data1)
        assay.add_sample_data(sample_data2)
        return assay


class TestAssayStorageSampleData(AssayStorageFixtures):
    def test_create_ok(self):
        memory.OnMemoryAssayStorage("test-assay", helpers.ConcreteRoi, helpers.ConcreteFeature)

    def test_has_sample_no_sample_returns_false(self, assay):
        assert not assay.has_sample("invalid-sample-id")

    def test_add_sample_data_ok(self, assay, sample_data1):
        assay.add_sample_data(sample_data1)
        assert assay.has_sample(sample_data1.get_sample().id)
        for roi in sample_data1.list_rois():
            assert assay.has_roi(roi.id)

    def test_add_sample_with_same_id_raises_error(self, assay, sample_data1):
        assay.add_sample_data(sample_data1)
        with pytest.raises(exceptions.RepeatedIdError):
            assay.add_sample_data(sample_data1)

    def test_has_sample(self, assay_with_sample_data, sample_data1):
        assert assay_with_sample_data.has_sample(sample_data1.get_sample().id)

    def test_fetch_sample(self, assay_with_sample_data, sample_data1):
        expected = sample_data1.get_sample()
        actual = assay_with_sample_data.fetch_sample(expected.id)
        assert actual == expected

    def test_fetch_sample_no_sample_raises_error(self, assay):
        with pytest.raises(exceptions.SampleNotFound):
            assay.fetch_sample("invalid-sample-id")

    def test_rois_from_sample_data_has_equal_id_in_assay_storage(self, assay_with_sample_data, sample_data1):
        for roi in sample_data1.list_rois():
            assert assay_with_sample_data.has_roi(roi.id)

    def test_fetch_rois_by_sample(self, assay_with_sample_data, sample_data1):
        expected = sample_data1.list_rois()
        actual = assay_with_sample_data.fetch_rois_by_sample(sample_data1.get_sample().id)
        assert actual
        assert all(x.equal(y) for x, y in zip(expected, actual))

    def test_fetch_rois_by_sample_invalid_sample_raises_error(self, assay_with_sample_data):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_sample_data.fetch_rois_by_sample("invalid-sample-id")

    def test_fetch_rois_by_sample_modifying_rois_do_not_modify_rois_in_storage(
        self, assay_with_sample_data, sample_data1
    ):
        modified = assay_with_sample_data.fetch_rois_by_sample(sample_data1.get_sample().id)[0]
        modified.data.append(1000.0)
        original = assay_with_sample_data.fetch_rois_by_id(modified.id)[0]
        assert not modified.equal(original)

    def test_fetch_rois_by_id_modifying_roi_do_not_modify_roi_in_storage(self, assay_with_sample_data, sample_data1):
        roi_id = sample_data1.list_rois()[0].id
        modified = assay_with_sample_data.fetch_rois_by_id(roi_id)[0]
        modified.data.append(1000.0)
        original = assay_with_sample_data.fetch_rois_by_id(roi_id)[0]
        assert not modified.equal(original)

    def test_fetch_roi_by_id_invalid_id_raises_error(self, assay):
        with pytest.raises(exceptions.RoiNotFound):
            assay.fetch_rois_by_id(create_id())

    def test_list_samples(self, assay_with_sample_data, sample_data1):
        expected = [sample_data1.get_sample()]
        actual = assay_with_sample_data.list_samples()
        assert actual == expected

    def test_fetch_features_by_id(self, assay_with_sample_data, sample_data1):
        expected = sample_data1.list_features()
        actual = assay_with_sample_data.fetch_features_by_id(*(x.id for x in expected))
        assert all(x.equal(y) for x, y in zip(actual, expected))

    def test_fetch_features_by_id_invalid_feature_id_raises_error(self, assay):
        with pytest.raises(exceptions.FeatureNotFound):
            assay.fetch_features_by_id(create_id())

    def test_fetch_features_by_sample_with_shared_roi_share_the_same_roi_instance(
        self, assay_with_sample_data, sample_data1
    ):
        _, ft2, ft3 = assay_with_sample_data.fetch_features_by_sample(sample_data1.get_sample().id)
        assert ft2.roi == ft3.roi
        assert ft2.roi is ft3.roi

    def test_fetch_features_by_group(self, assay_with_sample_data):
        group = 0
        actual = assay_with_sample_data.fetch_features_by_group(group)
        assert all(ft.annotation.group == group for ft in actual)

    def test_fetch_features_by_group_invalid_group_raises_error(self, assay_with_sample_data):
        with pytest.raises(exceptions.FeatureGroupNotFound):
            invalid_group = 1000
            assay_with_sample_data.fetch_features_by_group(invalid_group)


class TestOnMemoryAssayStorageDescriptorApi(AssayStorageFixtures):
    def test_fetch_feature_descriptors(self, assay_with_two_samples):
        descriptors = assay_with_two_samples.fetch_descriptors()
        descriptor_names = helpers.ConcreteFeature.descriptor_names()
        n_features = assay_with_two_samples.get_n_features()
        for name, values in descriptors.items():
            assert name in descriptor_names
            assert len(values) == n_features

    def test_fetch_features_descriptors_from_single_sample(self, assay_with_two_samples, sample_data1):
        sample_id = sample_data1.get_sample().id
        descriptors = assay_with_two_samples.fetch_descriptors(sample_id=sample_id)
        descriptor_names = helpers.ConcreteFeature.descriptor_names()
        n_features = sample_data1.get_n_features()
        for name, values in descriptors.items():
            assert name in descriptor_names
            assert len(values) == n_features

    def test_fetch_descriptors_subset(self, assay_with_two_samples):
        subset = {"mz", "height"}
        descriptors = assay_with_two_samples.fetch_descriptors(descriptors=subset)

        n_features = assay_with_two_samples.get_n_features()
        excluded_descriptor = "custom_descriptor"

        assert excluded_descriptor in helpers.ConcreteFeature.descriptor_names()
        assert excluded_descriptor not in descriptors

        for name, values in descriptors.items():
            assert name in subset
            assert len(values) == n_features

    def test_fetch_descriptors_subset_from_single_sample(self, assay_with_two_samples, sample_data1):
        subset = {"mz", "height"}
        sample_id = sample_data1.get_sample().id
        descriptors = assay_with_two_samples.fetch_descriptors(sample_id=sample_id, descriptors=subset)

        n_features = sample_data1.get_n_features()
        excluded_descriptor = "custom_descriptor"

        assert excluded_descriptor in helpers.ConcreteFeature.descriptor_names()
        assert excluded_descriptor not in descriptors

        for name, values in descriptors.items():
            assert name in subset
            assert len(values) == n_features

    def test_fetch_descriptors_invalid_sample_raises_error(self, assay_with_sample_data):
        with pytest.raises(exceptions.SampleNotFound):
            assay_with_sample_data.fetch_descriptors(sample_id="invalid-sample-id")

    def test_fetch_descriptors_invalid_descriptor_name_raises_error(self, assay_with_sample_data):
        with pytest.raises(exceptions.InvalidFeatureDescriptor):
            assay_with_sample_data.fetch_descriptors(descriptors=["invalid-descriptor"])

    def test_fetch_descriptors_retrieves_a_copy(self, assay_with_sample_data):
        descriptors = assay_with_sample_data.fetch_descriptors()

        # modify a random value in the descriptors table and check if values in the assay are modified
        name = list(descriptors)[0]
        original = descriptors[name][0]
        modified = 100000.0
        descriptors[name][0] = modified

        descriptors2 = assay_with_sample_data.fetch_descriptors()
        assert descriptors is not descriptors2
        assert descriptors != descriptors2
        assert descriptors2[name][0] == original

    def test_patch_descriptors_update_data(self, assay_with_sample_data):
        annotations = assay_with_sample_data.fetch_annotations()
        descriptors_before = assay_with_sample_data.fetch_descriptors()
        descriptor_name = "mz"  # arbitrary descriptor and index
        index = 2
        id_ = annotations[index].id

        original = descriptors_before[descriptor_name][index]
        modified = 100000.0
        patch = DescriptorPatch(id=id_, descriptor=descriptor_name, value=modified)

        assay_with_sample_data.patch_descriptors(patch)

        descriptors_after = assay_with_sample_data.fetch_descriptors()

        assert original != modified
        assert descriptors_before[descriptor_name][index] == original
        assert descriptors_after[descriptor_name][index] == modified

    def test_patch_descriptor_and_revert_to_previous_snapshot_retrieves_original_data(self, assay_with_sample_data):
        # first we create an snapshot
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)

        # then we apply a patch
        annotations = assay_with_sample_data.fetch_annotations()
        descriptors_before_patch = assay_with_sample_data.fetch_descriptors()
        index = 2
        id_ = annotations[index].id

        patch = DescriptorPatch(id=id_, descriptor="mz", value=100000.0)

        assay_with_sample_data.patch_descriptors(patch)
        descriptors_after_patch = assay_with_sample_data.fetch_descriptors()

        assay_with_sample_data.set_snapshot(snapshot_id)
        descriptors_snapshot = assay_with_sample_data.fetch_descriptors()

        assert descriptors_before_patch == descriptors_snapshot
        assert descriptors_before_patch != descriptors_after_patch

    def test_patch_descriptor_on_non_latest_snapshot_raises_error(self, assay_with_sample_data):
        # first we create an snapshot
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)
        assay_with_sample_data.set_snapshot(snapshot_id)

        # then we apply a patch
        annotations = assay_with_sample_data.fetch_annotations()
        index = 2
        id_ = annotations[index].id

        patch = DescriptorPatch(id=id_, descriptor="mz", value=100000.0)

        with pytest.raises(exceptions.SnapshotError):
            assay_with_sample_data.patch_descriptors(patch)

    def test_patch_descriptor_invalid_id_raises_error(self, assay_with_sample_data):
        invalid_id = create_id()
        patch = DescriptorPatch(id=invalid_id, descriptor="mz", value=100000.0)
        with pytest.raises(exceptions.FeatureNotFound):
            assay_with_sample_data.patch_descriptors(patch)


class TestOnMemoryAssayStorageAnnotationApi(AssayStorageFixtures):
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

    def test_fetch_annotations_retrieves_a_copy(self, assay_with_sample_data):
        annotations_before = assay_with_sample_data.fetch_annotations()
        index = 1
        annotations_before[index].group = 10000
        annotations_after = assay_with_sample_data.fetch_annotations()

        assert annotations_before != annotations_after

    def test_patch_annotations_update_data(self, assay_with_sample_data):
        annotations_before_patch = assay_with_sample_data.fetch_annotations()
        index = 1
        id_ = annotations_before_patch[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        assay_with_sample_data.patch_annotations(patch)
        annotations_after_patch = assay_with_sample_data.fetch_annotations()
        assert annotations_before_patch != annotations_after_patch
        assert annotations_after_patch[index].group == value

    def test_patch_annotations_and_revert_to_previous_snapshot_retrieves_original_data(self, assay_with_sample_data):
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)
        annotations_before_patch = assay_with_sample_data.fetch_annotations()
        index = 1
        id_ = annotations_before_patch[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        assay_with_sample_data.patch_annotations(patch)
        annotations_after_patch = assay_with_sample_data.fetch_annotations()

        assay_with_sample_data.set_snapshot(snapshot_id)
        annotations_snapshot = assay_with_sample_data.fetch_annotations()

        assert annotations_before_patch != annotations_after_patch
        assert annotations_before_patch == annotations_snapshot

    def test_patch_annotation_on_non_latest_snapshot_raises_error(self, assay_with_sample_data):
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)
        assay_with_sample_data.set_snapshot(snapshot_id)
        annotations = assay_with_sample_data.fetch_annotations()
        index = 1
        id_ = annotations[index].id
        field = "group"
        value = 1000
        patch = AnnotationPatch(id=id_, field=field, value=value)
        with pytest.raises(exceptions.SnapshotError):
            assay_with_sample_data.patch_annotations(patch)

    def test_patch_annotation_invalid_id_raises_error(self, assay_with_sample_data):
        invalid_id = create_id()
        patch = AnnotationPatch(id=invalid_id, field="group", value=10000)
        with pytest.raises(exceptions.FeatureNotFound):
            assay_with_sample_data.patch_annotations(patch)


class TestOnMemoryAssayStorageFillValueApi(AssayStorageFixtures):
    def test_fetch_fill_values_no_values_return_empty_dict(self, assay):
        fill_values = assay.fetch_fill_values()
        assert isinstance(fill_values, dict)
        assert not fill_values

    def test_fetch_add_values_and_fetch_return_same_fill_values(self, assay):
        val1 = FillValue(sample_id="sample1", feature_group=0, value=10.0)
        val2 = FillValue(sample_id="sample2", feature_group=1, value=5.0)
        assay.add_fill_values(val1, val2)
        actual = assay.fetch_fill_values()
        assert actual[val1.sample_id][val1.feature_group] == val1.value
        assert actual[val2.sample_id][val2.feature_group] == val2.value

    def test_fetch_fill_values_return_a_copy(self, assay):
        original = 10.0
        modified = 100.0
        val1 = FillValue(sample_id="sample1", feature_group=0, value=original)
        assay.add_fill_values(val1)
        fill_values_copy1 = assay.fetch_fill_values()
        fill_values_copy1[val1.sample_id][val1.feature_group] = modified
        fill_values_copy2 = assay.fetch_fill_values()
        assert fill_values_copy2[val1.sample_id][val1.feature_group] == original
        assert fill_values_copy1 is not fill_values_copy2

    def test_snapshots_maintain_independent_fill_values_copies(self, assay):
        # add fill values and create a snapshot
        val1 = FillValue(sample_id="sample1", feature_group=0, value=10.0)
        assay.add_fill_values(val1)
        snapshot_id = "snapshot"
        assay.create_snapshot(snapshot_id)

        # add more fill values and set previous snapshot
        val2 = FillValue(sample_id="sample2", feature_group=1, value=5.0)
        assay.add_fill_values(val2)
        assay.set_snapshot(snapshot_id)

        snapshot_fill_values = assay.fetch_fill_values()
        assert snapshot_fill_values[val1.sample_id][val1.feature_group] == val1.value
        assert val2.sample_id not in snapshot_fill_values

        # go back to latest status and check that value 2 is stored
        assay.set_snapshot()
        latest_fill_values = assay.fetch_fill_values()
        assert latest_fill_values[val1.sample_id][val1.feature_group] == val1.value
        assert latest_fill_values[val2.sample_id][val2.feature_group] == val2.value


class TestOnMemoryAssayStorageMiscApi(AssayStorageFixtures):
    def test_get_n_rois(self, assay_with_sample_data):
        expected = 2
        actual = assay_with_sample_data.get_n_rois()
        assert actual == expected

    def test_get_n_rois_empty_assay_returns_zero(self, assay):
        assert not assay.get_n_rois()

    def test_create_snapshot_using_reserved_id_raises_error(self, assay_with_sample_data):
        with pytest.raises(ValueError):
            assay_with_sample_data.create_snapshot(memory.LATEST)

    def test_create_snapshot_with_existing_id_raises_error(self, assay_with_sample_data):
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)
        with pytest.raises(exceptions.RepeatedIdError):
            assay_with_sample_data.create_snapshot(snapshot_id)

    def test_set_snapshot_invalid_id_raises_error(self, assay_with_sample_data):
        with pytest.raises(exceptions.SnapshotNotFound):
            assay_with_sample_data.set_snapshot("invalid-id")

    def test_set_snapshot_reset_deletes_snapshots(self, assay_with_sample_data):
        assay_with_sample_data.create_snapshot("snapshot1")
        assay_with_sample_data.create_snapshot("snapshot2")
        assay_with_sample_data.set_snapshot("snapshot1", reset=True)
        assert "snapshot2" not in assay_with_sample_data.list_snapshots()

    def test_set_reset_no_arguments_set_to_latest_snapshot(self, assay_with_sample_data):
        snapshot_id = "snapshot"
        assay_with_sample_data.create_snapshot(snapshot_id)
        assay_with_sample_data.set_snapshot(snapshot_id)
        assert assay_with_sample_data.get_snapshot_id() == snapshot_id
        assay_with_sample_data.set_snapshot()
        assert assay_with_sample_data.get_snapshot_id() == memory.LATEST
