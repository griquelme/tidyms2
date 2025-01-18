import pytest

from tidyms2.storage.memory import OnMemoryAssayStorage

from ... import helpers


@pytest.fixture
def assay_storage(sample_storage_with_features):
    assay = OnMemoryAssayStorage("assay", helpers.ConcreteRoi, helpers.ConcreteFeature)
    assay.add_sample_data(sample_storage_with_features)
    return assay


class TestAnnotationPatcher:
    @pytest.fixture
    def op(self):
        return helpers.DummyAnnotationPatcher(id="dummy-id")

    def test_apply_ok(self, op, assay_storage: OnMemoryAssayStorage):
        annotations_before = assay_storage.fetch_annotations()
        op.apply(assay_storage)
        annotations_after = assay_storage.fetch_annotations()
        assert annotations_before != annotations_after
        assert all(x.group == op.group for x in annotations_after)


class TestDescriptorPatcher:
    @pytest.fixture
    def op(self):
        return helpers.DummyDescriptorPatcher(id="dummy-id")

    def test_apply_ok(self, op, assay_storage: OnMemoryAssayStorage):
        descriptor = "custom_descriptor"
        descriptors_before = assay_storage.fetch_descriptors(descriptors=[descriptor])
        op.apply(assay_storage)
        descriptors_after = assay_storage.fetch_descriptors(descriptors=[descriptor])
        assert descriptors_before != descriptors_after
        assert all(x == op.patch for x in descriptors_after[descriptor])
